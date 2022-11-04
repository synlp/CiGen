import sys
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


gpu = 0
def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    lm_vocab = Vocab(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    # lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, 0.1, lm_args.approx)
    lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, 0.1)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.cuda(device)
    lm_model.eval()
    return lm_model, lm_vocab, lm_args


m_path = './model/songci.ckpt'
lm_model, lm_vocab, lm_args = init_model(m_path, gpu, "./model/vocab.txt")



ds = []
ds_ner = []
# with open("./data/dev.txt", "r") as f:
with open('./data/test.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            ds.append(line)
print(len(ds))
with open('./data/test_ner.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            ds_ner.append(line)
print(len(ds_ner))

local_rank = gpu
batch_size = 10
batches = round(len(ds) / batch_size)
idx = 0

avg_nll = 0.
avg_ppl = 0.
# avg_loss = 0.
count = 0.
while idx < len(ds):

    cplb = ds[idx:idx + batch_size]
    cplb_ner = ds_ner[idx:idx + batch_size]
    xs_tpl, xs_seg, xs_pos, \
    ys_truth, ys_inp, \
    ys_tpl, ys_seg, ys_pos, msk, _, _, _, ys_truth_ner = s2xy(cplb, cplb_ner, lm_vocab, lm_args.max_len, lm_args.min_len)

    xs_tpl = xs_tpl.cuda(local_rank)
    xs_seg = xs_seg.cuda(local_rank)
    xs_pos = xs_pos.cuda(local_rank)
    ys_truth = ys_truth.cuda(local_rank)
    ys_inp = ys_inp.cuda(local_rank)
    ys_tpl = ys_tpl.cuda(local_rank)
    ys_seg = ys_seg.cuda(local_rank)
    ys_pos = ys_pos.cuda(local_rank)
    msk = msk.cuda(local_rank)
    ys_truth_ner = ys_truth_ner.cuda(local_rank)

    # loss, bsz = lm_model.eval_loss(xs_tpl, xs_seg, xs_pos, ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk)
    nll, ppl, bsz = lm_model.ppl(xs_tpl, xs_seg, xs_pos, ys_truth, ys_tpl, ys_seg, ys_pos, msk, ys_truth_ner)
    # avg_loss += loss
    avg_nll += nll
    avg_ppl += ppl
    count += bsz

    idx += batch_size
    # if count % 200 == 0:
    #     print("nll=", avg_nll/count, "ppl=", avg_ppl/count, "count=", count)

# print("loss=", avg_loss/count, "count=", count)
print("nll=", avg_nll/count, "ppl=", avg_ppl/count, "count=", count)
with open('eval.log','w') as fout:
    fout.write(str(len(ds))+'\n')
    # fout.write('loss = %.2f, count = %d\n' % (avg_loss/count, count))
    fout.write('nll = %.2f, ppl = %.2f, count = %d\n' % (avg_nll/count, avg_ppl/count, count))