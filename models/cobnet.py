import torch
import numpy as np
import matplotlib.pyplot as plt
from models.cobnet_orientation import CobNetOrientationModule
from models.cobnet_fuse import CobNetFuseModule
from torch import nn
import utils as utls
from torchvision import transforms as trfms
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
import math


class CobNet(nn.Module):
    def __init__(self, n_orientations=8):

        super(CobNet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)

        self.reducers = nn.ModuleList([
            nn.Conv2d(self.base_model.conv1.out_channels,
                      out_channels=1,
                      kernel_size=1),
            nn.Conv2d(self.base_model.layer1[-1].conv3.out_channels,
                      out_channels=1,
                      kernel_size=1),
            nn.Conv2d(self.base_model.layer2[-1].conv3.out_channels,
                      out_channels=1,
                      kernel_size=1),
            nn.Conv2d(self.base_model.layer3[-1].conv3.out_channels,
                      out_channels=1,
                      kernel_size=1),
            nn.Conv2d(self.base_model.layer4[-1].conv3.out_channels,
                      out_channels=1,
                      kernel_size=1),
        ])

        for m in self.reducers:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)

        self.fuse = CobNetFuseModule()

        self.n_orientations = n_orientations
        self.orientations = nn.ModuleList(
            [CobNetOrientationModule() for _ in range(n_orientations)])

    def forward_sides(self, im):
        in_shape = im.shape[2:]
        # pass through base_model and store intermediate activations (sides)
        pre_sides = []
        x = self.base_model.conv1(im)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        pre_sides.append(x)
        x = self.base_model.layer1(x)
        pre_sides.append(x)
        x = self.base_model.layer2(x)
        pre_sides.append(x)
        x = self.base_model.layer3(x)
        pre_sides.append(x)
        x = self.base_model.layer4(x)
        pre_sides.append(x)

        late_sides = []
        upsamp = nn.UpsamplingBilinear2d(in_shape)
        for s, m in zip(pre_sides, self.reducers):
            late_sides.append(upsamp(m(s)))

        return pre_sides, late_sides

    def forward_orient(self, sides, shape=512):

        upsamp = nn.UpsamplingBilinear2d((shape, shape))
        orientations = []

        for m in self.orientations:
            or_ = upsamp(m(sides))
            orientations.append(or_)

        return orientations

    def forward_fuse(self, sides):

        y_fine, y_coarse = self.fuse(sides)

        return y_fine, y_coarse

    def forward(self, im):
        pre_sides, late_sides = self.forward_sides(im)

        orientations = self.forward_orient(pre_sides)
        y_fine, y_coarse = self.forward_fuse(late_sides)

        return {
            'pre_sides': pre_sides,
            'late_sides': late_sides,
            'orientations': orientations,
            'y_fine': y_fine,
            'y_coarse': y_coarse
        }
