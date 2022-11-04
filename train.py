# coding=utf-8
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from biglm import BIGLM
from data import Vocab, DataLoader, s2xy
from optim import Optim
from collections import  OrderedDict
import argparse, os
import random
import re

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--ff_embed_dim', type=int, default=3072)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.2)

    ###########
    parser.add_argument('--train_data_ner', type=str, default='./data/train_ner.txt')
    parser.add_argument('--dev_data_ner', type=str, default='./data/dev_ner.txt')
    ###########

    parser.add_argument('--train_data', type=str, default='./data/train.txt')
    parser.add_argument('--dev_data', type=str, default='./data/dev.txt')
    parser.add_argument('--vocab', type=str, default='./model/vocab.txt')
    parser.add_argument('--min_occur_cnt', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=8000)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--min_len', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--start_from', type=str, default='model/nat.ckpt')
    # parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='ckpt')

    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--MASTER_ADDR', type=str, default='localhost')
    parser.add_argument('--MASTER_PORT', type=str, default='28512')
    parser.add_argument('--start_rank', type=int, default=0)
    parser.add_argument('--backend', type=str, default='nccl')

    return parser.parse_args()

def update_lr(optimizer, lr): # 更新lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
 
def average_gradients(model):
    """ Gradient averaging. """
    normal = True
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
        else:
            normal = False
            break
    return normal

def eval_epoch(lm_args, model, lm_vocab, local_rank, label):
    print("validating...", flush=True)
    ds = []
    ds_ner = []
    with open(lm_args.dev_data, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ds.append(line)
    with open(lm_args.dev_data_ner, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ds_ner.append(line)

    batch_size = 10
    batches = round(len(ds) / batch_size)
    idx = 0
    avg_nll = 0.
    avg_ppl = 0.
    count = 0.
    while idx < len(ds):
        cplb = ds[idx:idx + batch_size]
        cplb_ner = ds_ner[idx:idx + batch_size]
        xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk, xs_tpl_all, xs_seg_all, xs_pos_all, ys_truth_ner = s2xy(cplb, cplb_ner, lm_vocab, lm_args.max_len, lm_args.min_len)

        xs_tpl = xs_tpl.cuda(local_rank)
        xs_seg = xs_seg.cuda(local_rank)
        xs_pos = xs_pos.cuda(local_rank)

        xs_tpl_all = xs_tpl_all.cuda(local_rank)
        xs_seg_all = xs_seg_all.cuda(local_rank)
        xs_pos_all = xs_pos_all.cuda(local_rank)

        ys_truth = ys_truth.cuda(local_rank)
        # ys_inp = ys_inp.cuda(local_rank)

        ys_truth_ner = ys_truth_ner.cuda(local_rank)

        ys_tpl = ys_tpl.cuda(local_rank)
        ys_seg = ys_seg.cuda(local_rank)
        ys_pos = ys_pos.cuda(local_rank)
        msk = msk.cuda(local_rank)

        nll, ppl, bsz = model.ppl(xs_tpl, xs_seg, xs_pos, ys_truth, ys_tpl, ys_seg, ys_pos, msk, ys_truth_ner)
        avg_nll += nll
        avg_ppl += ppl
        count += bsz

        idx += batch_size

    print(label, "nll=", avg_nll / count, "ppl=", avg_ppl / count, "count=", count, flush=True)
    return avg_ppl / count

def run(args, local_rank):
    """ Distributed Synchronous """
    torch.manual_seed(1234)
    vocab = Vocab(args.vocab, min_occur_cnt=args.min_occur_cnt, specials=[])
    if (args.world_size == 1 or dist.get_rank() == 0):
        print ("vocab.size = " + str(vocab.size), flush=True)
    model = BIGLM(local_rank, vocab, args.embed_dim, args.ff_embed_dim,\
                  args.num_heads, args.dropout, args.layers, args.smoothing)

    print(model)
    if args.start_from is not None:
        ckpt = torch.load(args.start_from, map_location='cpu')
        # old_state_dict = ckpt['model']
        # new_state_dict = OrderedDict()
        # for k, v in old_state_dict.items():
        #     k = 'model_lm.'+k
        #     new_state_dict[k] = v
        # model.load_state_dict(new_state_dict)
        model.load_state_dict(ckpt['model'], strict=False)
    model = model.cuda(local_rank)

    # param_optimizer = list(model.named_parameters())
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not re.findall('model_lm', n)]}
    # ] # num_of_params = 256 (the same for model_lm)
    # optimizer = Optim(model.embed_dim, args.lr, args.warmup_steps,
    #                   torch.optim.Adam(optimizer_grouped_parameters, lr=0, betas=(0.9, 0.998), eps=1e-9))

    optimizer = Optim(model.embed_dim, args.lr, args.warmup_steps, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.998), eps=1e-9))

    # if args.start_from is not None:
    #     optimizer.load_state_dict(ckpt['optimizer'])

    # train_data = DataLoader(vocab, args.train_data, args.batch_size, args.max_len, args.min_len)
    train_data = DataLoader(vocab, args.train_data, args.train_data_ner, args.batch_size, args.max_len, args.min_len)
    batch_acm = 0
    acc_acm, nll_acm, ppl_acm, ntokens_acm, nxs, npairs_acm, loss_acm = 0., 0., 0., 0., 0., 0., 0.
    format_greedy_acm, format_sample_acm, loss_rl_acm, loss_nat_acm = 0., 0., 0., 0.
    loss_nat_stage1_acm, loss_nat_stage2_acm = 0., 0.
    best_ppl = 100000.
    while True:
        model.train()
        if train_data.epoch_id > 30:
            break
        for xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk, xs_tpl_all, xs_seg_all, xs_pos_all, ys_truth_ner in train_data:
            batch_acm += 1
            xs_tpl = xs_tpl.cuda(local_rank)
            xs_seg = xs_seg.cuda(local_rank)
            xs_pos = xs_pos.cuda(local_rank)

            xs_tpl_all = xs_tpl_all.cuda(local_rank)
            xs_seg_all = xs_seg_all.cuda(local_rank)
            xs_pos_all = xs_pos_all.cuda(local_rank)

            ys_truth = ys_truth.cuda(local_rank)
            # ys_inp = ys_inp.cuda(local_rank)

            ys_truth_ner = ys_truth_ner.cuda(local_rank)

            ys_tpl = ys_tpl.cuda(local_rank)
            ys_seg = ys_seg.cuda(local_rank)
            ys_pos = ys_pos.cuda(local_rank)
            msk = msk.cuda(local_rank)

            model.zero_grad()
            # res, loss, acc, nll, ppl, ntokens, npairs = model.forward_rl_with_lm(xs_tpl, xs_seg, xs_pos, ys_truth, ys_tpl, ys_seg, ys_pos, msk, xs_tpl_all, xs_seg_all, xs_pos_all)

            res, loss, acc, nll, ppl, ntokens, npairs, format_greedy, format_sample, loss_rl, loss_nat_stage1, loss_nat_stage2, loss_nat = model.forward_rl_with_lm(xs_tpl, xs_seg, xs_pos,
                                                                                 ys_tpl, ys_seg, ys_pos, msk,
                                                                                 xs_tpl_all, xs_seg_all, xs_pos_all, ys_truth, ys_truth_ner)

            # res, loss, acc, nll, ppl, ntokens, npairs, format_greedy, format_sample, loss_rl, loss_nat_stage1, loss_nat_stage2, loss_nat = model.forward_nat(
            #     xs_tpl, xs_seg, xs_pos,
            #     ys_tpl, ys_seg, ys_pos, msk,
            #     xs_tpl_all, xs_seg_all, xs_pos_all, ys_truth, ys_truth_ner)



            loss_acm += loss.item()
            acc_acm += acc
            nll_acm += nll
            ppl_acm += ppl
            ntokens_acm += ntokens
            npairs_acm += npairs
            nxs += npairs

            try:
                format_greedy_acm += format_greedy.item()
                format_sample_acm += format_sample.item()
            except:
                format_greedy_acm += format_greedy
                format_sample_acm += format_sample
            try:
                loss_rl_acm += loss_rl.item()
            except:
                loss_rl_acm += loss_rl
            loss_nat_acm += loss_nat
            loss_nat_stage1_acm += loss_nat_stage1
            loss_nat_stage2_acm += loss_nat_stage2
            
            loss.backward()
            if args.world_size > 1:
                is_normal = average_gradients(model)
            else:
                is_normal = True
            if is_normal:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                print("gradient: none, gpu: " + str(local_rank), flush=True)
                continue

            if (args.world_size == 1 or dist.get_rank() == 0) and batch_acm % args.print_every == -1 % args.print_every:
                print('batch_acm %d, loss %.3f, loss_rl %.3f, loss_nat_stage1 %.3f, loss_nat_stage2 %.3f, loss_nat %.3f, format_greedy %.4f, format_sample %.4f acc %.3f, nll %.3f, ppl %.3f, x_acm %d, lr %.6f' \
                      % (batch_acm, loss_acm / args.print_every, \
                         loss_rl_acm / args.print_every, loss_nat_stage1_acm / args.print_every, loss_nat_stage2_acm / args.print_every, loss_nat_acm / args.print_every, \
                         format_greedy_acm / args.print_every, format_sample_acm / args.print_every, \
                         acc_acm / ntokens_acm, \
                         nll_acm / nxs, ppl_acm / nxs, npairs_acm, optimizer._rate), flush=True)
                acc_acm, nll_acm, ppl_acm, ntokens_acm, loss_acm, nxs = 0., 0., 0., 0., 0., 0.
                format_greedy_acm, format_sample_acm, loss_rl_acm, loss_nat_acm = 0., 0., 0., 0.
                loss_nat_stage1_acm, loss_nat_stage2_acm = 0., 0.
            if (args.world_size == 1 or dist.get_rank() == 0) and batch_acm % args.save_every == -1 % args.save_every:
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)

                # model.eval()
                # ppl_before = eval_epoch(args, model, vocab, local_rank, "epoch-" + str(train_data.epoch_id) + "-acm-" + str(batch_acm))
                # model.train()

                torch.save({'args':args, 'model':model.state_dict()}, '%s/epoch%d_batch_%d'%(args.save_dir, train_data.epoch_id, batch_acm))
                # if ppl_before < best_ppl:
                    # torch.save({'args':args, 'model':model.state_dict(), 'optimizer':optimizer.state_dict()}, '%s/songci.ckpt'%(args.save_dir))
                    # best_ppl = ppl_before

def init_processes(args, local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank + local_rank, world_size=args.world_size)
    fn(args, local_rank)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_config()

    if args.world_size == 1:
        run(args, 0)
        exit(0)
    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=init_processes, args=(args, rank, run, args.backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
