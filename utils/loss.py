#!/usr/bin/env python3
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BalancedBCE(nn.Module):
    def __init__(self):
        super(BalancedBCE, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input,
                                                 target,
                                                 reduction='none')

        bsize = target.shape[0]
        beta_pos = torch.tensor([1 - t.sum() / t.numel() for t in target])
        beta_neg = torch.tensor([t.sum() / t.numel() for t in target])
        loss_pos = torch.cat([
            beta_pos[i] * bce[i][target[i] == 1] for i in range(bsize)
        ]).mean()
        loss_neg = torch.cat([
            beta_neg[i] * bce[i][target[i] == 0] for i in range(bsize)
        ]).mean()

        return loss_pos + loss_neg
