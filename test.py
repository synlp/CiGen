import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy 
import time
import os
from biglm import BIGLM
from data import Vocab, DataLoader, s2t, s2xy
import sys

sys_arg = str(sys.argv[2])


def init_seeds():
    random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

#init_seeds()

gpu = 0
def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    lm_vocab = Vocab(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, 0.1)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.cuda(device)
    lm_model.eval()
    return lm_model, lm_vocab, lm_args

# m_path = "./model/songci.ckpt"
m_path = str(sys.argv[1])
lm_model, lm_vocab, lm_args = init_model(m_path, gpu, "./model/vocab.txt")


k = 32

def top_k(xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk):
    with torch.no_grad():
        k=32
        start = time.time()
        probs, logprobs = lm_model.work(xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk)

        tmp = torch.empty_like(logprobs).fill_(float('-inf'))
        topk, indices = torch.topk(logprobs, 32, dim=2)
        tmp = tmp.scatter(2, indices, topk)
        gen_sample = torch.distributions.Categorical(logits=tmp.detach()).sample()

        res = lm_vocab.idx2token(gen_sample[:, 0].cpu().numpy().tolist())
        print(time.time() - start)
        return res


ds = []
with open("./data/test.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            ds.append(line)
print(len(ds))

local_rank = gpu
batch_size = 1
# cp_size = 1
# batches = round(len(ds) / batch_size)

for i in range(5): 
    idx = 0
    if not os.path.exists("./results_all/results_"+sys_arg+"/top-"+str(k)):
        os.makedirs("./results_all/results_"+sys_arg+"/top-"+str(k))
    with open("./results_all/results_"+sys_arg+"/top-"+str(k)+"/out"+str(i+1)+".txt", "w") as fo:
        while idx < len(ds):
            cplb = ds[idx:idx + batch_size]
            # cplb = []
            # for line in lb:
            #     cplb += [line for i in range(cp_size)]
            print(cplb)
            xs_tpl, xs_seg, xs_pos, \
            ys_truth, ys_inp, \
            ys_tpl, ys_seg, ys_pos, msk, _, _, _, _ = s2xy(cplb, cplb, lm_vocab, lm_args.max_len, 2)

            xs_tpl = xs_tpl.cuda(local_rank)
            xs_seg = xs_seg.cuda(local_rank)
            xs_pos = xs_pos.cuda(local_rank)
            ys_tpl = ys_tpl.cuda(local_rank)
            ys_seg = ys_seg.cuda(local_rank)
            ys_pos = ys_pos.cuda(local_rank)

            msk = msk.cuda(local_rank)

            # enc, src_padding_mask = lm_model.encode(xs_tpl, xs_seg, xs_pos)
            # s = [['<bos>']] * batch_size
            # res = top_k(enc, src_padding_mask, ys_tpl, ys_seg, ys_pos)

            res = top_k(xs_tpl, xs_seg, xs_pos, ys_tpl, ys_seg, ys_pos, msk)

            # for i, line in enumerate(cplb):
            #     r = ''.join(res[i])
            #     print(line)
            #     print(r)
            print(cplb[0])
            print(''.join(res))
            fo.write(cplb[0] + "\t" + ''.join(res) + "\n")

            idx += batch_size

