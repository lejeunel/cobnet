import torch
import numpy as np
import matplotlib.pyplot as plt
from cobnet_orientation import CobNetOrientationModule
from cobnet_fuse import CobNetFuseModule
from torch import nn
import utils as utls
from torchvision import transforms as trfms
import torchvision.models as models
from torchvision.models.resnet import Bottleneck


def remove_bn(network):
    for layer in network.children():
        if isinstance(layer, nn.Sequential) or (isinstance(layer, Bottleneck)):
            remove_bn(layer)
        if list(layer.children()) == []:  # if leaf node, add it to list
            if isinstance(layer, nn.BatchNorm2d):
                layer = nn.Identity()
    return network


# Model for Convolutional Oriented Boundaries
# Needs a base model (vgg, resnet, ...) from which intermediate
# features are extracted
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

        self.fuse = CobNetFuseModule()

        self.n_orientations = n_orientations
        self.orientations = nn.ModuleList(
            [CobNetOrientationModule() for _ in range(n_orientations)])

    def forward(self, im):

        in_shape = im.shape[2:]
        # pass through base_model and store intermediate activations (sides)
        sides = []
        x = self.base_model.conv1(im)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        sides.append(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        sides.append(x)
        x = self.base_model.layer2(x)
        sides.append(x)
        x = self.base_model.layer3(x)
        sides.append(x)
        x = self.base_model.layer4(x)
        sides.append(x)

        reduced_sides = []
        upsamp = nn.UpsamplingBilinear2d(in_shape)
        for s, m in zip(sides, self.reducers):
            reduced_sides.append(upsamp(m(s)))
        cat_sides = torch.cat(reduced_sides, dim=1)

        y_fine, y_coarse = self.fuse(cat_sides)

        orientations = []
        for m in self.orientations:
            or_ = upsamp(m(sides))
            orientations.append(or_)

        return {
            'y_fine': y_fine,
            'y_coarse': y_coarse,
            'sides': reduced_sides,
            'orientations': orientations
        }
