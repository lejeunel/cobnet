#!/usr/bin/env python3
import torch
from torch import nn
import torch.nn.functional as F


class BalancedBCE(nn.Module):
    def __init__(self, cuda=True):
        super(BalancedBCE, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input,
                                                 target,
                                                 reduction='none')

        beta = target.sum() / target.numel()
        if target.sum() > 0:
            loss = ((1 - beta) * bce[target == 1]).mean()
            loss += (beta * bce[target == 0]).mean()
        else:
            loss = bce.mean()

        return loss
