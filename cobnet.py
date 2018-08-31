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
from cobnet_orientation import CobNetOrientationModule
from cobnet_fuse import CobNetFuseModule

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
                 batch_size=4,
                 shuffle=True,
                 num_workers=0,
                 num_epochs=20,
                 lr=0.001,
                 momentum=0.9,
                 cuda=False):

        super(CobNet, self).__init__()
        self.base_model = MyResnet50(cuda=cuda, batch_size=batch_size)

        # We add a "weight-fusion" layer that sum outputs
        # of all side outputs to produce a global edge map
        # Is there a bias term in xie et. al 2015?
        self.fuse_model = CobNetFuseModule(self.base_model, (224, 224))

        # Image preprocessing
        # Trained on ImageNet where images are normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
        # We use the same normalization statistics here.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))])

        self.n_orientations = n_orientations # Parameter K in maninis17
        self.max_angle_orientations = max_angle_orientations


        # Parameter M in maninis17
        self.n_inputs_per_sub_nets = n_inputs_per_sub_nets

        self.orientation_keys = np.linspace(0, self.max_angle_orientations,
                                            self.n_orientations, dtype=int)

        self.orientation_modules = self.make_all_orientation_modules()

        self.dataloader_train = CobDataLoader(img_paths_train,
                                              truth_paths_train,
                                              self.transform)

        self.dataloader_val = CobDataLoader(img_paths_val,
                                            truth_paths_val,
                                            self.transform)

        self.dataloaders = {'train': DataLoader(self.dataloader_train,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers),
                            'val': DataLoader(self.dataloader_val,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)}

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

        return dict_

    def deepcopy(self):

        dict_ = {k:copy.deepcopy(v)
                 for k,v in self.get_all_modules_as_dict().items()}
        return dict_

    def train_mode(self):

        self.base_model.train()

        for _,m in self.orientation_modules.items():
            m.train()

    def eval_mode(self):

        for name, module in self.get_all_modules_as_dict().items():
            module.eval()

    def make_all_orientation_modules(self):
        # Build dictionaries of per-orientation modules

        models_orient = dict()
        for orient in self.orientation_keys:
            m_ = CobNetOrientationModule(self.base_model,
                                         orient,
                                         (224, 224))
            models_orient[orient] = m_

        return models_orient


    def forward(self, im):
        # im: Input image (Tensor)
        # target_sides: Truths for all side outputs (ResNet part)
        # target_orient: Truths for all orientations (Orientation modules)

        # Pass through resnet
        self.base_model(im)

        # Store tensors in dictionary
        y_sides = {l:self.base_model.output_tensor(l) for l in range(5)}

        y_fuse = self.fuse_model(y_sides)

        #y_orient = dict() # This stores one output per orientation module
        #for orient in self.orientation_keys:
        #    y_orient[orient] = self.orientation_modules[orient](outputs)

        return y_sides, y_fuse

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


    def forward(self, y_sides, y_fused,
                target_sides=None,
                target_fused=None):
        # im: Input image (Tensor)
        # target_sides: Truths for all scales (ResNet part)
        # target_orient: Truths for all orientations (Orientation modules)


        # make transforms to resize target to size of y_sides[s]
        import pdb; pdb.set_trace()
        resize_transf = {s: transforms.Resize(y_sides[s].shape[-2:])
                         for s in y_sides.keys()}
        loss_sides = 0
        if(target_sides is not None):
            for s in y_sides.keys():
                # beta is for class imbalance
                t_  = resize_transf[s](target_sides[s])
                beta = torch.sum(t_)/np.prod(y_sides[s].shape)
                idx_pos = torch.where(t_ > 0)
                idx_neg = torch.where(t_ == 0)
                loss_scales += -beta*torch.sum(y_sides[s][idx_pos]) \
                    -(1-beta)*torch.sum(y_sides[s][idx_neg])

        # Compute loss for fused edge map
        loss_fuse = 0
        # make transforms to resize target to size of y_fused
        resize_transf = transforms.Resize(y_fused.shape[-2:])

        if(target_fused is not None):
            # beta is for class imbalance
            t_  = resize_transf(target_fused)
            beta = torch.sum(t_)/np.prod(y_fused.shape)
            idx_pos = torch.where(t_ > 0)
            idx_neg = torch.where(t_ == 0)
            loss_fuse = -beta*torch.sum(y_fused[idx_pos]) \
                -(1-beta)*torch.sum(y_fused[idx_neg])


        #y_orient = dict() # This stores one output per orientation module
        #for orient in self.orientation_keys:
        #    y_orient[orient] = self.forward_on_module(y_scale, im, orient)

        return loss_sides + loss_fuse
