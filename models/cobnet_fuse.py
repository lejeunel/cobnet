import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
import os
import math


class CobNetFuseModule(nn.Module):
    """
    This performs a linear weighting of side activations
    to return a fine and coarse edge map
    """
    def __init__(self, n_sides=4, init_prob=0.1):
        super(CobNetFuseModule, self).__init__()
        self.fine = nn.Conv2d(n_sides, 1, kernel_size=1)
        self.coarse = nn.Conv2d(n_sides, 1, kernel_size=1)

        # bias = -math.log((1 - init_prob) / init_prob)
        # self.fine.bias.data.fill_(bias)
        # self.coarse.bias.data.fill_(bias)
        nn.init.constant_(self.fine.weight, 0.25)
        nn.init.constant_(self.coarse.weight, 0.25)

    def get_bias(self):
        return [self.fine.bias, self.coarse.bias]

    def get_weight(self):
        return [self.fine.weight, self.coarse.weight]

    def forward(self, sides):
        y_fine = self.fine(torch.cat(sides[:4], dim=1))
        y_coarse = self.coarse(torch.cat(sides[1:], dim=1))

        return y_fine, y_coarse
