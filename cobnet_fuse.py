import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
import os

class CobNetFuseModule(nn.Module):
    def __init__(self, base_model, out_shape, batch_size=4, cuda=True):

        super(CobNetFuseModule, self).__init__()

        self.base_model = base_model

        self.out_shape = out_shape
        self.transform =  transforms.Compose([
            transforms.Resize(self.out_shape),
            transforms.ToPILImage(self.out_shape)])

        self.batch_size = batch_size

        self.side_crop = [None, 1, 2, 4 ,8]
        self.deconv_kernels = [None, 4, 8, 16, 32]
        self.deconv_strides = [None, 2, 4, 8, 16]

        # Build it
        self.side_models, self.fusion_model = self.make_model()

        if(cuda):
            for m in self.side_models:
                for l in m:
                    if(isinstance(l, nn.Module)):
                        l.cuda()
            self.fusion_model.cuda()

    def get_params(self):

        # side_models layers
        params = list()
        for m in self.side_models:
            for l in m:
                if(isinstance(l, nn.Conv2d)):
                    params.append({'params': [l.weight],
                               'lr': 1e-2,
                               'lr_decay': 1})
                    params.append({'params': [l.bias],
                               'lr': 2e-2,
                               'lr_decay': 0})

        params.append({'params': [self.fusion_model[0].weight],
                   'lr': 1e-2,
                   'lr_decay': 1})

        return params

    def load(self, path):

        for i,m in enumerate(self.side_models):
            for l in m:
                if(isinstance(l, nn.Conv2d)):
                    l.load_state_dict(
                        torch.load(
                            os.path.join(path,'side_{}.pth'.format(i))))

        self.fusion_model.load_state_dict(
            torch.load(
                os.path.join(path,'fusion.pth')))

    def load_state_dict(self, dict_):

        for i,m in enumerate(self.side_models):
            for l in m:
                if(isinstance(l, nn.Conv2d)):
                    l.load_state_dict(dict_['side'][i])

        self.fusion_model.load_state_dict(dict_['fusion'])

    def state_dict(self):

        out = dict()
        out['side'] = {k: s[0].state_dict()
                       for k,s in enumerate(self.side_models)}

        out['fusion'] = self.fusion_model.state_dict()

        return out

    def save(self, path):

        for i,m in enumerate(self.side_models):
            for l in m:
                if(isinstance(l, nn.Conv2d)):
                    torch.save(l.state_dict(),
                               os.path.join(path, 'side_{}.pth'.format(i)))

        torch.save(self.fusion_model.state_dict(),
                   os.path.join(path, 'fusion.pth'))

    def train(self):

        for m in self.side_models:
            for l in m:
                if(isinstance(l, nn.Module)):
                    l.train()

        self.fusion_model.train()

    def eval(self):

        for m in self.side_models:
            for l in m:
                if(isinstance(l, nn.Module)):
                    l.eval()

        self.fusion_model.eval()
        
    def forward(self, x):

        A_side = dict()
        Y_side = dict()

        # Convolve ,deconvolve, and crop each side output
        for i,m in enumerate(self.side_models):
            A_side[i] = m[0](x[i])
            A_side[i] = m[1](A_side[i])
            A_side[i] = m[2](A_side[i])

        # Make Y_side, probability maps of each side outputs
        for i,a in A_side.items():
            Y_side[i] = nn.Sigmoid()(a).squeeze(1)

        # Concatenate side outputs
        cat_ = torch.cat([v for k,v in A_side.items()],
                         dim=1)

        # Pass into fusion model
        Y_fused = self.fusion_model(cat_).squeeze(1)

        return Y_side, Y_fused

    def make_model(self):

        def crop(x, c):
            return x[..., c:-c, c:-c]

        def dummy(x):
            return x

        model = list()

        # Convolution layers to make resnet outputs of depth 1
        conv_layers = [
            nn.Conv2d(in_channels=self.base_model.output_tensor_shape(i)[1],
                      out_channels=1,
                      kernel_size=1,
                      stride=1)
            for i in range(5)]

        # Upsample side outputs to original image size
        sizes_sides = [self.base_model.output_tensor_shape(i)[2:]
                       for i in range(5)]
        scale_factors = [(self.out_shape[0]//s[0], self.out_shape[1]//s[1], 1)
                         for s in sizes_sides]

        deconv_layers = dict()
        deconv_layers[0] = dummy
        deconv_layers[1] = nn.ConvTranspose2d(1, 1,
                                              kernel_size=self.deconv_kernels[1],
                                              stride=self.deconv_strides[1])
        deconv_layers[2] = nn.ConvTranspose2d(1, 1,
                                              kernel_size=self.deconv_kernels[2],
                                              stride=self.deconv_strides[2])
        deconv_layers[3] = nn.ConvTranspose2d(1, 1,
                                              kernel_size=self.deconv_kernels[3],
                                              stride=self.deconv_strides[3])
        deconv_layers[4] = nn.ConvTranspose2d(1, 1,
                                              kernel_size=self.deconv_kernels[4],
                                              stride=self.deconv_strides[4])
        deconv_layers = [v for _,v in deconv_layers.items()]

        crop_layers = dict()
        crop_layers[0] = dummy
        crop_layers[1] = lambda x: crop(x, self.side_crop[1])
        crop_layers[2] = lambda x: crop(x, self.side_crop[2])
        crop_layers[3] = lambda x: crop(x, self.side_crop[3])
        crop_layers[4] = lambda x: crop(x, self.side_crop[4])
        crop_layers = [v for _,v in crop_layers.items()]

        side_models = [[a, b, c]
                        for a, b, c
                       in zip(conv_layers, deconv_layers, crop_layers)]

        fusion_model = nn.Sequential(*[nn.Conv2d(in_channels=5,
                                               out_channels=1,
                                               kernel_size=1,
                                               stride=1),
                                     nn.Sigmoid()])

        return side_models, fusion_model

