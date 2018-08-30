import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.models import VGG
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import copy
from myresnet50 import MyResnet50
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from dataloader import CobDataLoader
from cobnet_orient import CobNetOrientationModule

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
                 num_workers=0,
                 num_epochs=20,
                 lr=0.001,
                 momentum=0.9,
                 cuda=False):

        super(CobNet, self).__init__()
        self.base_model = MyResnet50(cuda=False, batch_size=batch_size)

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
                                              transform)

        self.dataloader_val = CobDataLoader(img_paths_val,
                                            truth_paths_val,
                                            transform)

        self.dataloaders = {'train': DataLoader(self.dataloader_train,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers),
                            'val': DataLoader(self.dataloader_val,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)}
        for i, sample in enumerate(self.dataloader_train):
            print(sample[0].size(), sample[1].size())

            if i == 3:
                break

        for i, sample in enumerate(self.dataloader_val):
            print(sample[0].size(), sample[1].size())

            if i == 3:
                break

        self.device = torch.device("cuda:0" if cuda \
                                   else "cpu")

        self.criterion = CobNetLoss()
        self.num_epochs = num_epochs
        self.num_works = num_workers
        self.lr = lr
        self.momentum = momentum

    def get_all_modules_as_dict(self):

        dict_ = dict()
        dict_['base_model'] = self.base_model
        dict_['orientation_modules'] = self.orientation_modules
        dict_['scale_modules'] = self.scale_modules

        return dict_

    def deepcopy(self):

        dict_ = {k:copy.deepcopy(v)
                 for k,v in self.get_all_modules_as_dict().items()}
        return dict_

    def train_mode(self):

        self.base_model.train()

        for m in self.orientation_modules:
            m.train()

        for k, v in self.scale_modules.items():
            v.train()

    def eval_mode(self):

        for name, module in self.get_all_modules_as_dict().items():
            module.eval()

    def make_all_output_orientation_modules(self):
        # Build for each orientation the output module,
        # i.e. conv2d + sigmoid

        return None

    def make_all_orientation_modules(self):
        # Build dictionaries of per-orientation modules

        models_orient = dict()
        for orient in self.orientation_keys:
            m_ = CobNetOrientationModule(self.base_model, orient)
            models_orient[orient] = m_

        return models_orient

    def make_scale_layers(self):
        # Build dictionaries of per-scale sigmoid layers
        # These layers are supervised individually
        
        modules = dict()
        for s in range(5):
            modules[s] = nn.Sequential(*[nn.Sigmoid()])

        return modules


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

        y_orient = dict() # This stores one output per orientation module
        for orient in self.orientation_keys:
            y_orient[orient] = self.orientation_modules[orient](im)

        return y_scale, y_orient

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

        best_model_wts = self.deepcopy()
        best_acc = 0.0

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(self.base_model.parameters(),
                              lr=self.lr,
                              momentum=self.momentum)

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer,
                                               step_size=7,
                                               gamma=0.1)

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
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
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.forward(inputs)
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


    def forward(self, y_scale, target_scale=None):
        # im: Input image (Tensor)
        # target_scale: Truths for all scales (ResNet part)
        # target_orient: Truths for all orientations (Orientation modules)


        # make transforms to resize target to size of y_scale[s]
        resize_transf = {s: transforms.Resize(y_scale[s].shape[-2:])
                         for s in y_scale.keys()}
        import pdb; pdb.set_trace()
        loss_scales = 0
        if(target_scale is not None):
            for s in y_scale.keys():
                # beta is for class imbalance
                t_  = resize_transf[s](target_scale[s])
                beta = torch.sum(t_)/np.prod(y_scale[s].shape)
                idx_pos = torch.where(t_ > 0)
                idx_neg = torch.where(t_ == 0)
                loss_scales += -beta*torch.sum(y_scale[s][idx_pos]) \
                    -(1-beta)*torch.sum(y_scale[s][idx_neg])


        #y_orient = dict() # This stores one output per orientation module
        #for orient in self.orientation_keys:
        #    y_orient[orient] = self.forward_on_module(y_scale, im, orient)

        return loss_scales
