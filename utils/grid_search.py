# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, os

lrs = np.logspace(-7,-3,10)
bws = np.logspace(-4,1,10)

for lr in lrs:
    for bw in bws:
        cmd = 'CUDA_VISIBLE_DEVICES=2 python semi-main2,py --lr %f --b_weight %f'%(lr,bw)


