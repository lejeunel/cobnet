#!/usr/bin/env python3
import torch
from torch import nn
import torch.nn.functional as F


class BalancedBCE(nn.Module):
    def __init__(self, cuda=True):
        super(BalancedBCE, self).__init__()

    def forward(self, input, target):
        beta = target.sum() / target.numel()
        bce = F.binary_cross_entropy_with_logits(input,
                                                 target,
                                                 reduction='none')
        bce_pos = bce[target == 1].mean()
        bce_neg = bce[target == 0].mean()

        loss = (1 - beta) * bce_pos + beta * bce_neg

        return loss
