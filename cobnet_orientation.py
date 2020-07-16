import torch.nn as nn
import torch


class CobNetOrientationModule(nn.Module):
    def __init__(self, in_channels=[64, 256, 512, 1024, 2048]):

        super(CobNetOrientationModule, self).__init__()

        # From model:
        # https://github.com/kmaninis/COB/blob/master/models/deploy.prototxt

        self.stages = nn.ModuleList()
        for i, inc in enumerate(in_channels):
            module = []
            module.append(nn.Conv2d(inc, 32, kernel_size=3, padding=1))
            module.append(nn.Conv2d(32, 4, kernel_size=3, padding=1))

            self.stages.append(nn.Sequential(*module))

        self.last_conv = nn.Conv2d(20, 1, kernel_size=3, padding=1)

    def forward(self, sides):

        x = []
        upsamp = nn.UpsamplingBilinear2d(sides[0].shape[2:])
        for m, s in zip(self.stages, sides):
            x.append(upsamp(m(s)))

        # concatenate all modules and merge
        x = torch.cat(x, dim=1)
        x = self.last_conv(x)

        return x
