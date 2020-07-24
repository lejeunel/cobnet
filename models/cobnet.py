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
from torch.nn import functional as F


def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1:y1 + th, x1:x1 + tw]


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 -
                                                 abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


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
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

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
        pre_sides.append(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        pre_sides.append(x)
        x = self.base_model.layer2(x)
        pre_sides.append(x)
        x = self.base_model.layer3(x)
        pre_sides.append(x)
        x = self.base_model.layer4(x)
        pre_sides.append(x)

        late_sides = []
        for s, m in zip(pre_sides, self.reducers):
            late_sides.append(m(s))

        # img_H, img_W = in_shape[0], in_shape[1]
        # weight_deconv0 = make_bilinear_weights(2, 1).cuda()
        # weight_deconv1 = make_bilinear_weights(4, 1).cuda()
        # weight_deconv2 = make_bilinear_weights(8, 1).cuda()
        # weight_deconv3 = make_bilinear_weights(16, 1).cuda()
        # weight_deconv4 = make_bilinear_weights(32, 1).cuda()

        # upsample0 = F.conv_transpose2d(late_sides[0], weight_deconv0, stride=2)
        # upsample1 = F.conv_transpose2d(late_sides[1], weight_deconv1, stride=4)
        # upsample2 = F.conv_transpose2d(late_sides[2], weight_deconv2, stride=8)
        # upsample3 = F.conv_transpose2d(late_sides[3],
        #                                weight_deconv3,
        #                                stride=16)
        # upsample4 = F.conv_transpose2d(late_sides[4],
        #                                weight_deconv4,
        #                                stride=32)

        # so0 = crop(upsample0, img_H, img_W)
        # so1 = crop(upsample1, img_H, img_W)
        # so2 = crop(upsample2, img_H, img_W)
        # so3 = crop(upsample3, img_H, img_W)
        # so4 = crop(upsample4, img_H, img_W)
        upsamp = nn.UpsamplingBilinear2d(in_shape)
        so0 = upsamp(late_sides[0])
        so1 = upsamp(late_sides[1])
        so2 = upsamp(late_sides[2])
        so3 = upsamp(late_sides[3])
        so4 = upsamp(late_sides[4])

        return pre_sides, [so0, so1, so2, so3, so4]

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
