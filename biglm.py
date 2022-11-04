import torch
from torch import nn
import torch.nn.functional as F

from utils import gelu, LayerNorm
from transformer import TransformerLayer, Embedding, LearnedPositionalEmbedding, SelfAttentionMask
from label_smoothing import LabelSmoothing 
from LanguageModelEval.lm_eval import init_model
from metrics import eval_tpl, eval_rhythm
import re


class BIGLM(nn.Module):
    def __init__(self, local_rank, vocab, embed_dim, ff_embed_dim, num_heads, dropout, layers, smoothing_factor, approx=None):
        super(BIGLM, self).__init__()
        self.vocab = vocab 
        self.embed_dim = embed_dim 

        self.tok_embed = Embedding(self.vocab.size, embed_dim, self.vocab.padding_idx) 
        self.pos_embed = LearnedPositionalEmbedding(embed_dim, device=local_rank) 
        
        self.layers_stage1 = nn.ModuleList() 
        self.layers_stage2 = nn.ModuleList() 
        for i in range(int(layers/2)): 
            self.layers_stage1.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout, with_external=True))
            self.layers_stage2.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout, with_external=True))
        self.emb_layer_norm = LayerNorm(embed_dim) 

        self.one_more_stage1 = nn.Linear(embed_dim, embed_dim) 
        self.one_more_layer_norm_stage1 = LayerNorm(embed_dim) 
        self.out_proj_stage1 = nn.Linear(embed_dim, self.vocab.size)

        self.one_more_stage2 = nn.Linear(embed_dim, embed_dim) 
        self.one_more_layer_norm_stage2 = LayerNorm(embed_dim) 
        self.out_proj_stage2 = nn.Linear(embed_dim, self.vocab.size)
        
        self.attn_mask = SelfAttentionMask(device=local_rank)
        self.smoothing = LabelSmoothing(local_rank, self.vocab.size, self.vocab.padding_idx, smoothing_factor)
       
        self.dropout = dropout
        self.device = local_rank

        self.approx = approx
        self.reset_parameters() 

        

    def reset_parameters(self): 
        nn.init.constant_(self.one_more_stage1.bias, 0.)
        nn.init.normal_(self.one_more_stage1.weight, std=0.02)
        nn.init.constant_(self.out_proj_stage1.bias, 0.)
        nn.init.normal_(self.out_proj_stage1.weight, std=0.02)
        nn.init.constant_(self.one_more_stage2.bias, 0.)
        nn.init.normal_(self.one_more_stage2.weight, std=0.02)
        nn.init.constant_(self.out_proj_stage2.bias, 0.)
        nn.init.normal_(self.out_proj_stage2.weight, std=0.02)
    
    def label_smotthing_loss(self, y_pred, y, y_mask, avg=True):
        seq_len, bsz = y.size()

        y_pred = torch.log(y_pred.clamp(min=1e-8))
        loss = self.smoothing(y_pred.view(seq_len * bsz, -1), y.view(seq_len * bsz, -1))
        if avg:
            return loss / torch.sum(y_mask)
        else:
            return loss / bsz

    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -torch.log(torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        ppl = 2 ** cost
        return cost.sum().item(), ppl.sum().item()

 

    def work(self, xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk):
        seq_len, bsz = ys_tpl.size()
        with torch.no_grad():
            probs, logprobs = self.forward_base_stage1(xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk)
            
            _, gen_sample = probs.max(-1)

            tmp0 = torch.eq(gen_sample, self.vocab.token2idx('_'))
            tmp0 = tmp0.unsqueeze(-1)

            inp = gen_sample.clone()
            inp = torch.cat([torch.ones((1, inp.size(1))).to(inp) * self.vocab.token2idx('<bos>'), inp], 0)
            tmp = torch.eq(inp, self.vocab.token2idx('<eos>'))
            inp = ~tmp * inp 
            
            probs1 = ~tmp0 * probs
            logprobs1 = ~tmp0 * logprobs

            ###########################stage 2
            probs, logprobs = self.forward_base_stage2(xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk, ys_inp_stage1=inp[:-1], stage=2)

            probs2 = tmp0 * probs
            logprobs2 = tmp0 * logprobs

            probs = probs1 + probs2
            logprobs = logprobs1 + logprobs2

            return probs, logprobs
    
    def encode(self, xs_tpl, xs_seg, xs_pos):
        padding_mask = torch.eq(xs_tpl, self.vocab.padding_idx)
        x = self.tok_embed(xs_tpl)  + self.tok_embed(xs_seg) + self.tok_embed(xs_pos)
        x = self.emb_layer_norm(x)
        return x, padding_mask

    def ppl(self, xs_tpl, xs_seg, xs_pos, ys_truth, ys_tpl, ys_seg, ys_pos, msk, ys_truth_ner):
        seq_len, bsz = ys_tpl.size()
        with torch.no_grad():
            probs, logprobs = self.forward_base(xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk, ys_truth=ys_truth_ner)
            

            shifted = ys_truth_ner.clone()
            shifted = torch.cat([torch.ones((1, ys_truth_ner.size(1))).to(shifted) * self.vocab.token2idx('<bos>'), shifted], 0)[:-1]
            tmp = torch.eq(shifted, self.vocab.token2idx('<eos>'))
            shifted = ~tmp * shifted 
            tmp = torch.eq(ys_truth_ner, self.vocab.token2idx('_'))
            tmp = tmp.unsqueeze(-1)
            probs1 = tmp * probs

            ###########################stage 2
            probs, logprobs = self.forward_base(xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk, ys_inp_stage1=shifted, ys_truth=ys_truth, stage=2)

            probs2 = ~tmp * probs
            probs = probs1 + probs2

            nll, ppl = self.nll_loss(probs, ys_truth, msk)
            return nll, ppl, bsz


    def forward_base_stage1(self, xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk, ys_inp_stage1=None, ys_truth=None, stage=1):
        enc, src_padding_mask = self.encode(xs_tpl, xs_seg, xs_pos)
        seq_len, bsz = ys_tpl.size()
        x = self.pos_embed(ys_tpl) + self.tok_embed(ys_tpl) + self.tok_embed(ys_seg) + self.tok_embed(ys_pos)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(ys_tpl, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers_stage1:
            x, _, _ = layer(x, self_padding_mask=padding_mask, \
                            self_attn_mask=None, \
                            external_memories=enc, \
                            external_padding_mask=src_padding_mask, )

        x = self.one_more_layer_norm_stage1(gelu(self.one_more_stage1(x)))
        logits = self.out_proj_stage1(x)
        probs = torch.softmax(logits, -1)
        logprobs = torch.log_softmax(logits, -1)

        
        return probs, logprobs

    def forward_base_stage2(self, xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk, ys_inp_stage1=None, ys_truth=None, stage=1):
        enc, src_padding_mask = self.encode(xs_tpl, xs_seg, xs_pos)
        seq_len, bsz = ys_tpl.size()
        x = self.pos_embed(ys_tpl) + self.tok_embed(ys_tpl) + self.tok_embed(ys_seg) + self.tok_embed(ys_pos) + self.tok_embed(ys_inp_stage1)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(ys_tpl, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers_stage2:
            x, _, _ = layer(x, self_padding_mask=padding_mask, \
                            self_attn_mask=None, \
                            external_memories=enc, \
                            external_padding_mask=src_padding_mask, )

        x = self.one_more_layer_norm_stage2(gelu(self.one_more_stage2(x)))
        logits = self.out_proj_stage2(x)
        probs = torch.softmax(logits, -1)
        logprobs = torch.log_softmax(logits, -1)

        
        return probs, logprobs


    def forward_rl_with_lm(self, xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk, xs_tpl_all, xs_seg_all, xs_pos_all, ys_truth=None, ys_truth_ner=None):
        seq_len, bsz = ys_tpl.size()
        with torch.no_grad():
            ###########################stage 1
            probs_greedy, logprobs_greedy = self.forward_base_stage1(xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk, ys_truth=ys_truth_ner)
            
            _, gen_greedy = logprobs_greedy.max(-1)

            
            tmp0 = torch.eq(gen_greedy, self.vocab.token2idx('_'))
            tmp0 = tmp0.unsqueeze(-1)

            inp = gen_greedy.clone()
            inp = torch.cat([torch.ones((1, inp.size(1))).to(inp) * self.vocab.token2idx('<bos>'), inp], 0)
            tmp = torch.eq(inp, self.vocab.token2idx('<eos>'))
            inp = ~tmp * inp

            logprobs_greedy1 = ~tmp0 * logprobs_greedy

            _, logprobs_greedy = self.forward_base_stage2(xs_tpl, xs_seg, xs_pos,  ys_tpl, ys_seg, ys_pos, msk, ys_inp_stage1=inp[:-1], ys_truth=ys_truth, stage=2)

            logprobs_greedy2 = tmp0 * logprobs_greedy
            logprobs_greedy = logprobs_greedy1 + logprobs_greedy2

            _, gen_greedy = logprobs_greedy.max(-1)

            format_greedy = self.compute_ppl_with_lm(xs_tpl_all, xs_seg_all, xs_pos_all, gen_greedy, ys_tpl, ys_seg, ys_pos, msk, ys_truth)

        ################################stage 1
        probs, logprobs = self.forward_base_stage1(xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk, ys_truth=ys_truth_ner)
        
        _, gen_sample = probs.max(-1)
        
        tmp0 = torch.eq(gen_sample, self.vocab.token2idx('_'))
        tmp0 = tmp0.unsqueeze(-1)
        
        probs1 = ~tmp0 * probs
        logprobs1 = ~tmp0 * logprobs
        inp = gen_sample.clone()
        inp = torch.cat([torch.ones((1, inp.size(1))).to(inp) * self.vocab.token2idx('<bos>'), inp], 0)
        tmp = torch.eq(inp, self.vocab.token2idx('<eos>'))
        inp = ~tmp * inp

        loss_nat_stage1 = self.label_smotthing_loss(probs, ys_truth_ner, msk)
        ###########################stage 2
        
        ys_truth_stage2 = tmp0.squeeze(-1) * ys_truth
        tmp1 = torch.ones(ys_truth.size(), device=ys_truth.device, dtype=torch.int)
        tmp1 = ~tmp0.squeeze(-1) * tmp1
        tmp1 = self.vocab.token2idx('_') * tmp1
        ys_truth_stage2 += tmp1


        probs, logprobs = self.forward_base_stage2(xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk, ys_inp_stage1=inp[:-1], ys_truth=ys_truth_stage2, stage=2)
        
        probs2 = tmp0 * probs
        logprobs2 = tmp0 * logprobs

        tmp2 = probs2.clone()
        tmp2[:,:,self.vocab.token2idx('_')] = 1.
        tmp2 = ~tmp0 * tmp2

        probs2 += tmp2

        loss_nat_stage2 = self.label_smotthing_loss(probs2, ys_truth_stage2, msk)


        logprobs = logprobs1 + logprobs2
        tmp = torch.empty_like(logprobs).fill_(float('-inf'))
        topk, indices = torch.topk(logprobs, 32, dim=2)
        tmp = tmp.scatter(2, indices, topk)
        gen_sample = torch.distributions.Categorical(logits=tmp.detach()).sample()
        
        format_sample = self.compute_ppl_with_lm(xs_tpl_all, xs_seg_all, xs_pos_all, gen_sample, ys_tpl, ys_seg, ys_pos, msk, ys_truth)
        

        reward_ppl = format_greedy - format_sample

        loss_rl = self.rl_criterion(logprobs, gen_sample, reward_ppl, msk)

        loss_nat = loss_nat_stage1 + loss_nat_stage2

        loss = loss_nat + loss_rl
        
        _, pred_y = logprobs.max(-1)
        tot_tokens = msk.float().sum().item()
        acc = (torch.eq(pred_y, ys_truth).float() * msk).sum().item()

        nll, ppl = self.nll_loss(torch.exp(logprobs), ys_truth, msk)

        
        return (pred_y, ys_truth), loss, acc, nll, ppl, tot_tokens, bsz, torch.mean(format_greedy), torch.mean(format_sample), loss_rl, loss_nat_stage1.item(), loss_nat_stage2.item(), loss_nat.item()

    
    def compute_ppl_with_lm(self, xs_tpl, xs_seg, xs_pos, ys_truth, ys_tpl, ys_seg, ys_pos, msk, ys_truth_orig):
        with torch.no_grad():
            tpl = []
            rhythm = []
            micro_dist1 = []
            micro_dist2 = []
            for i in range(xs_tpl.size(1)):
                sents1 = (''.join(self.vocab.idx2token(ys_truth_orig[:,i].cpu().numpy().tolist()))).split('<eos>')[0].split('</s>')[:-1]
                sents2 = (''.join(self.vocab.idx2token(ys_truth[:,i].cpu().numpy().tolist()))).split('<eos>')[0].split('</s>')[:-1]
                sents2 = list(filter(None, sents2))
                p, r, f1, n0, n1, n2 = eval_tpl(sents1, sents2)
                try:
                    tpl_micro_p = n0 / n2
                    tpl_micro_r = n0 / n1
                    tpl_micro_f1 = 2 * tpl_micro_p * tpl_micro_r / (tpl_micro_p + tpl_micro_r)
                except:
                    tpl_micro_f1 = 0
                tpl.append(tpl_micro_f1)

                p, r, f1, n0, n1, n2 = eval_rhythm(sents1, sents2)
                try:
                    rhy_micro_p = n0 / n2
                    rhy_micro_r = n0 / n1
                    rhy_micro_f1 = 2 * rhy_micro_p * rhy_micro_r / (rhy_micro_p + rhy_micro_r)
                except:
                    rhy_micro_f1 = 0
                rhythm.append(rhy_micro_f1)
                ugrams = [w for w in ''.join(sents2)]
                bigrams = []
                for bi in range(len(ugrams) - 1):
                    bigrams.append(ugrams[bi] + ugrams[bi + 1])
                try:
                    d1 = len(set(ugrams)) / float(len(ugrams))
                except:
                    d1 = 0
                try:
                    d2 = len(set(bigrams)) / float(len(bigrams))
                except:
                    d2 = 0
                micro_dist1.append(d1)
                micro_dist2.append(d2)

            tpl = torch.tensor(tpl).unsqueeze(0).T.cuda(xs_tpl.device.index)
            rhythm = torch.tensor(rhythm).unsqueeze(0).T.cuda(xs_tpl.device.index)
            micro_dist1 = torch.tensor(micro_dist1).unsqueeze(0).T.cuda(xs_tpl.device.index)
            micro_dist2 = torch.tensor(micro_dist2).unsqueeze(0).T.cuda(xs_tpl.device.index)
            format_score = (tpl+rhythm+micro_dist1+micro_dist2)/4
            
        return format_score

    def rl_criterion(self, input, seq, reward, mask):
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        input = input.reshape(-1)
        reward = reward.squeeze(-1).unsqueeze(0).repeat(seq.size(0), 1).reshape(-1)
        mask = mask.reshape(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

