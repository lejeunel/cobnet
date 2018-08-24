import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.models import VGG
from torchvision import transforms
import os
from myresnet50 import MyResnet50
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from dataloader import CobDataLoader

# Model for Convolutional Oriented Boundaries
# Needs a base model (vgg, resnet, ...) from which intermediate
# features are extracted
class CobNet(nn.Module):
    def __init__(self, n_orientations=8,
                 max_angle_orientations=140, # in degrees
                 n_inputs_per_sub_nets=5,
                 img_paths_train=None,
                 truth_paths_train=None,
                 img_paths_val=None,
                 truth_paths_val=None,
                 transform=None,
                 batch_size=4,
                 shuffle=True,
                 num_workers=4,
                 num_epochs=20,
                 cuda=False):

        super(CobNet, self).__init__()
        self.base_model = MyResnet50(cuda=False)

        self.n_orientations = n_orientations # Parameter K in maninis17
        self.max_angle_orientations = max_angle_orientations


        # Parameter M in maninis17
        self.n_inputs_per_sub_nets = n_inputs_per_sub_nets

        self.orientation_keys = np.linspace(0, self.max_angle_orientations,
                                            self.n_orientations, dtype=int)

        self.orientation_modules = self.make_all_orientation_modules()
        self.output_modules = self.make_all_output_orientation_modules()
        self.scale_modules = self.make_scale_layers()

        self.dataloader_train = CobDataLoader(img_paths_train,
                                        truth_paths_train,
                                        transform,
                                        batch_size,
                                        shuffle,
                                        num_workers)

        self.dataloader_val = CobDataLoader(img_paths_val,
                                        truth_paths_val,
                                        transform,
                                        batch_size,
                                        shuffle,
                                        num_workers)

        self.dataloaders = {'train': self.dataloader_train,
                            'val': self.dataloader_val}

        self.device = torch.device("cuda:0" if cuda \
                                   else "cpu")

        self.criterion = CobNetLoss()

    def get_all_modules_as_dict(self):

        dict_ = dict()
        dict_['base_model'] = self.base_model
        dict_['orientation_modules'] = self.orientation_modules
        dict_['scale_modules'] = self.scale_modules

        return dict_

    def deepcopy(self):

        dict_ = {k:copy.deepcopy(v) for k,v in self.get_all_modules_as_dict()}
        return dict_

    def train_mode(self):

        for name, module in self.get_all_modules_as_dict().items():
            module.train()

    def eval_mode(self):

        for name, module in self.get_all_modules_as_dict().items():
            module.eval()

    def make_all_output_orientation_modules(self):
        # Build for each orientation the output module,
        # i.e. conv2d + sigmoid

        modules = dict()
        for orient in self.orientation_keys:
            modules[orient] = nn.Sequential(*[nn.Conv2d(
                in_channels=7*5, # 7 dims per scale
                out_channels=1,
                padding=1,
                kernel_size=3),
                                              nn.Sigmoid()])

        return modules

    def make_all_orientation_modules(self):
        # Build dictionaries of per-orientation modules

        modules = dict()
        for orient in self.orientation_keys:
            modules[orient] = self.make_single_orientation_layers()

        return modules

    def make_scale_layers(self):
        # Build dictionaries of per-scale sigmoid layers
        # These layers are supervised individually
        
        modules = dict()
        for s in range(5):
            modules[s] = nn.Sequential(*[nn.Sigmoid()])

        return modules

    def make_single_orientation_layers(self):
        # From model:
        # https://github.com/kmaninis/COB/blob/master/models/deploy.prototxt

        # Scale 1 (Will take conv1 layer as input)
        conv_1 = nn.Conv2d(
            in_channels=self.base_model.model.conv1.out_channels,
            out_channels=32,
            kernel_size=3,
            padding=1)
        conv_2 = nn.Conv2d(
            in_channels=conv_1.out_channels,
            out_channels=4,
            kernel_size=3,
            padding=1)

        # Scale 2 (Will take conv3 of last bottleneck as input)
        conv_3 = nn.Conv2d(
            in_channels=self.base_model.model.layer1.__getitem__(-1).conv3.out_channels,
            out_channels=4,
            kernel_size=3,
            padding=1)
        conv_4 = nn.Conv2d(
            in_channels=conv_3.out_channels,
            out_channels=4,
            kernel_size=3,
            padding=1)
        deconv_1 = nn.ConvTranspose2d(
            in_channels=conv_4.out_channels,
            out_channels=4,
            stride=2,
            kernel_size=4)

        ## Scale 3 (Will take conv3 of last bottleneck as input)
        conv_5 = nn.Conv2d(
            in_channels=self.base_model.model.layer2.__getitem__(-1).conv3.out_channels,
            out_channels=32,
            kernel_size=3,
            padding=1)
        conv_6 = nn.Conv2d(
            in_channels=conv_5.out_channels,
            out_channels=4,
            kernel_size=3,
            padding=1)
        deconv_2 = nn.ConvTranspose2d(
            in_channels=conv_6.out_channels,
            out_channels=4,
            kernel_size=8,
            stride=4)
        ## crop concat???

        ## Scale 4 (Will take conv3 of last bottleneck as input)
        conv_7 = nn.Conv2d(
            in_channels=self.base_model.model.layer3.__getitem__(-1).conv3.out_channels,
            out_channels=32,
            kernel_size=3,
            padding=1)
        conv_8 = nn.Conv2d(
            in_channels=conv_7.out_channels,
            out_channels=4,
            kernel_size=3,
            padding=1)
        deconv_3 = nn.ConvTranspose2d(
            in_channels=conv_8.out_channels,
            out_channels=4,
            kernel_size=16,
            stride=8)
        ## crop concat???

        ## Scale 5
        conv_9 = nn.Conv2d(
            in_channels=self.base_model.model.layer4.__getitem__(-1).conv3.out_channels,
            out_channels=32,
            kernel_size=3,
            padding=1)
        conv_10 = nn.Conv2d(
            in_channels=conv_7.out_channels,
            out_channels=4,
            kernel_size=3,
            padding=1)
        deconv_4 = nn.ConvTranspose2d(
            in_channels=conv_10.out_channels,
            out_channels=4,
            kernel_size=32,
            stride=16)
        ## crop concat???

        self.conv_layers = [conv_1, conv_2, conv_3,
                            conv_4, conv_5, conv_6,
                            conv_7, conv_8, conv_9, conv_10]

        for l in self.conv_layers:
            nn.init.normal_(l.weight, mean=0, std=0.01)

        # Make dict for each scale
        dict_ = {0: nn.Sequential(*[conv_1, conv_2]),
                 1: nn.Sequential(*[conv_3, conv_4, deconv_1]),
                 2: nn.Sequential(*[conv_5, conv_6, deconv_2]),
                 3: nn.Sequential(*[conv_7, conv_8, deconv_3]),
                 4: nn.Sequential(*[conv_9, conv_10, deconv_4])}

        return dict_

    def forward(self, im):
        # im: Input image (Tensor)
        # target_scale: Truths for all scales (ResNet part)
        # target_orient: Truths for all orientations (Orientation modules)

        # Pass through resnet
        outputs = self.base_model(im)

        # Get tensors at output layers for each scale
        # Each output is supervised separately
        conv1_out = outputs[0]
        layer1_out = outputs[1]
        layer2_out = outputs[2]
        layer3_out = outputs[3]
        layer4_out = outputs[4]

        # Store tensors in dictionary
        y_scale = {0: self.scale_modules[0](conv1_out),
                   1: self.scale_modules[1](layer1_out),
                   2: self.scale_modules[2](layer2_out),
                   3: self.scale_modules[3](layer3_out),
                   4: self.scale_modules[4](layer4_out)}

        #y_orient = dict() # This stores one output per orientation module
        #for orient in self.orientation_keys:
        #    y_orient[orient] = self.forward_on_module(y_scale, im, orient)

        return y_scale

    def forward_on_module(self, x, im, orientation):
        # Forward pass of dict of tensors x.
        # Input image (PIL) im is concatenated
        # Orientation is a key for orientation [0, 20, ..., 140]

        mod_ = self.orientation_modules[orientation]
        out_mod_ = self.output_modules[orientation]


        width, height = im.shape[2:]
        transform = transforms.Compose([
            transforms.ToPILImage(),
                    transforms.Resize((width//2, height//2)),
                    transforms.ToTensor()])

        im_resized = transform(im.squeeze(0)).unsqueeze(0)
        width, height = im_resized.shape[2:]

        # Forward pass on each layer and concat input image
        # Keys are in 1, ..., 5
        y = dict()
        for key in x.keys():
            y[key] = mod_[key](x[key])

            # crop response
            y_width = y[key].shape[-2]
            y_height = y[key].shape[-1]
            half_crop_size = (y_width - width)//2
            y_resized = Variable(y[key])[:,:,
                                         half_crop_size:y_width-half_crop_size,
                                         half_crop_size:y_width-half_crop_size]

            # Concatenate image tensor
            y[key] = torch.cat((im_resized, y_resized), 1)


        #Resize all to input image size
        # Concatenate outputs of all modules
        y_all = torch.cat([y[k] for k in y.keys()], 1)
        y_all = out_mod_(y_all) # conv2d + sigmoid

        return y_all

    def train(self):
        since = time.time()

        import pdb; pdb.set_trace()
        best_model_wts = self.deepcopy()
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.train_mode()  # Set model to training mode
                else:
                    self.eval_mode()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

class CobNetLoss(nn.Module):

    def __init__(self):
        super(CobNetLoss, self).__init__()


    def forward(self, out_scales, target_scale=None):
        # im: Input image (Tensor)
        # target_scale: Truths for all scales (ResNet part)
        # target_orient: Truths for all orientations (Orientation modules)

        loss_scales = 0
        if(target_scale is not None):
            # class imbalance
            for s in range(5):
                beta = torch.sum(target_scale[s])/np.prod(y_scale[s].shape)
                idx_pos = torch.where(target_scale[s] > 0)
                idx_neg = torch.where(target_scale[s] == 0)
                loss_scales += -beta*torch.sum(y_scale[s][idx_pos]) \
                    -(1-beta)*torch.sum(y_scale[s][idx_neg])


        #y_orient = dict() # This stores one output per orientation module
        #for orient in self.orientation_keys:
        #    y_orient[orient] = self.forward_on_module(y_scale, im, orient)

        return loss_scales
