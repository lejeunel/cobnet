import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
import os


class CobNetFuseModule(nn.Module):
    def __init__(self, n_sides=4):

        super(CobNetFuseModule, self).__init__()
        self.fine = nn.Conv2d(n_sides, 1, kernel_size=1)
        self.coarse = nn.Conv2d(n_sides, 1, kernel_size=1)

    def forward(self, x):
        y_fine = self.fine(x[:, :4, ...])
        y_coarse = self.coarse(x[:, 1:, ...])

        return y_fine, y_coarse
