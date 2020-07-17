import torch
import torchvision
from torch import nn
import numpy as np

x = torch.zeros((1,1,7,7))
deconv = nn.ConvTranspose2d(1, 1, kernel_size=32, stride=16)
y = deconv(x)

print('x.shape: {}'.format(x.shape))
print('y.shape: {}'.format(y.shape))

def dummy(t):
    return t

print('dummy(x).shape: {}'.format(dummy(x).shape))

z = torch.ones((1, 1, 512, 512))
print('z.shape: {}'.format(z.shape))

out = z.sum()
print(z.div(out))
