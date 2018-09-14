import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import copy
from myresnet50 import MyResnet50
import numpy as np
import matplotlib.pyplot as plt
from dataloader import CobDataLoader
from cobnet_orientation import CobNetOrientationModule
from cobnet_fuse import CobNetFuseModule
import utils as utls
from loss_logger import LossLogger


# Model for Convolutional Oriented Boundaries
# Needs a base model (vgg, resnet, ...) from which intermediate
# features are extracted
class CobNet(nn.Module):
    def __init__(
            self,
            n_orientations=8,
            max_angle_orientations=140,  # in degrees
            n_inputs_per_sub_nets=5,
            img_paths_train=None,
            truth_paths_train=None,
            img_paths_val=None,
            truth_paths_val=None,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            num_epochs=100,
            lr=1 * 10e-5,
            weight_decay=2 * 10e-4,
            momentum=0.9,
            lr_switch=1 * 10e-6,  # This lr kicks in after epoch_switch
            num_epochs_switch=90,
            cuda=False,
            save_path='checkpoints'):

        super(CobNet, self).__init__()
        self.base_model = MyResnet50(cuda=cuda, batch_size=batch_size)

        # Image preprocessing
        # Trained on ImageNet where images are normalized
        # We use the same normalization statistics here.
        self.batch_size = batch_size
        self.base_shape = (224, 224)
        self.base_mean_norm = [0.485, 0.456, 0.406]
        self.base_std_norm = [0.229, 0.224, 0.225]
        self.save_path = save_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.num_works = num_workers
        self.lr = lr
        self.momentum = momentum
        self.lr_switch=lr_switch
        self.num_epochs_switch = num_epochs_switch

        # We add a "weight-fusion" layer that sum outputs
        # of all side outputs to produce a global edge map
        # Is there a bias term in xie et. al 2015?
        self.fuse_model = CobNetFuseModule(
            self.base_model, np.asarray(self.base_shape) // 2, cuda=cuda)

        self.shapes_of_sides = [
            self.base_model.output_tensor_shape(i)[-2:] for i in range(5)
        ]

        self.n_orientations = n_orientations  # Parameter K in maninis17
        self.max_angle_orientations = max_angle_orientations

        # Parameter M in maninis17
        self.n_inputs_per_sub_nets = n_inputs_per_sub_nets

        self.orientation_keys = np.linspace(
            0, self.max_angle_orientations, self.n_orientations, dtype=int)

        self.orientation_modules = self.make_all_orientation_modules()

        self.dataloader_train = CobDataLoader(
            img_paths_train,
            self.shapes_of_sides,
            self.base_shape,
            self.base_mean_norm,
            self.base_std_norm,
            truth_paths=truth_paths_train)

        self.dataloader_val = CobDataLoader(
            img_paths_val,
            self.shapes_of_sides,
            self.base_shape,
            self.base_mean_norm,
            self.base_std_norm,
            truth_paths=truth_paths_val)

        self.dataloaders = {
            'train':
            DataLoader(
                self.dataloader_train,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers),
            'val':
            DataLoader(
                self.dataloader_val,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers)
        }

        self.device = torch.device("cuda:0" if cuda \
                                   else "cpu")

        self.criterion = CobNetLoss(cuda=cuda)

    def get_all_modules_as_dict(self):

        dict_ = dict()
        dict_['base_model'] = self.base_model
        dict_['fuse_model'] = self.fuse_model
        #dict_['orientation_modules'] = self.orientation_modules

        return dict_

    def deepcopy(self):

        dict_ = {
            k: copy.deepcopy(v)
            for k, v in self.get_all_modules_as_dict().items()
        }
        return dict_

    def load_state_dict(self, dict_):

        self.base_model.load_state_dict(dict_['resnet'])
        self.fuse_model.load_state_dict(dict_['fuse'])

    def state_dict(self):

        return {
            'resnet': self.base_model.state_dict(),
            'fuse': self.fuse_model.state_dict()
        }

    def load(self, path):

        self.base_model.load(path)
        self.fuse_model.load(path)

    def save(self, save_dir):

        if (not os.path.exists(save_dir)):
            os.makedirs(save_dir)

        self.base_model.save(save_dir)
        self.fuse_model.save(save_dir)

    def train_mode(self):

        self.base_model.train()

        for _, m in self.orientation_modules.items():
            m.train()

    def eval_mode(self):

        for name, module in self.get_all_modules_as_dict().items():
            module.eval()

    def make_all_orientation_modules(self):
        # Build dictionaries of per-orientation modules

        models_orient = dict()
        for orient in self.orientation_keys:
            m_ = CobNetOrientationModule(self.base_model, orient, (224, 224))
            models_orient[orient] = m_

        return models_orient

    def forward(self, im):
        # im: Input image (Tensor)
        # target_sides: Truths for all side outputs (ResNet part)
        # target_orient: Truths for all orientations (Orientation modules)

        # Pass through resnet
        self.base_model(im)

        # Store tensors in dictionary
        x_sides = {l: self.base_model.output_tensor(l) for l in range(5)}

        Y_sides, Y_fused = self.fuse_model(x_sides)

        #y_orient = dict() # This stores one output per orientation module
        #for orient in self.orientation_keys:
        #    y_orient[orient] = self.orientation_modules[orient](outputs)

        return Y_sides, Y_fused

    def get_params(self):

        params = [self.fuse_model.get_params(), self.base_model.get_params()]

        return params[0] + params[1]

    def train(self):
        since = time.time()

        best_model_wts = self.deepcopy()
        best_val_loss = float("inf")

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(
            self.get_params(),
            momentum=self.momentum,
            lr=self.lr,
            weight_decay=self.weight_decay)

        train_logger = LossLogger('train', self.batch_size,
                                  len(self.dataloaders['train']),
                                  self.save_path)

        val_logger = LossLogger('val', self.batch_size,
                                len(self.dataloaders['val']), self.save_path)

        loggers = {'train': train_logger, 'val': val_logger}

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # lower learning rate...
            if(epoch > self.num_epochs_switch):
                new_params = utls.set_optim_params_sgd(
                    optimizer.state_dict()['param_groups'],
                    lr=self.lr_switch)
                optimizer.state_dict()['param_groups'] = new_params

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    #scheduler.step()
                    self.train_mode()  # Set model to training mode
                else:
                    self.eval_mode()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                samp = 1
                for inputs, labels_sides, labels_fuse in self.dataloaders[
                        phase]:

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs_sides, outputs_fuse = self.forward(inputs)
                        loss = self.criterion(outputs_sides, outputs_fuse,
                                              labels_sides, labels_fuse)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    loggers[phase].update(epoch, samp,
                                          loss.item() * inputs.size(0))

                    samp += 1

                loggers[phase].print_epoch(epoch)

                # Generate train prediction for check
                if phase == 'train':
                    path = os.path.join(self.save_path, 'previews',
                                        'epoch_{}.jpg'.format(epoch))
                    _, pred = self.forward(inputs)
                    im_ = inputs[0, ...].detach().cpu()
                    im_ = self.dataloader_train.im_inv_transform(im_)
                    im_ = np.asarray(im_)
                    utls.save_tensors(im_, pred, labels_fuse[0, ...], path)

                # deep copy the model
                if phase == 'val' and (loggers['val'].get_loss(epoch) <
                                       best_val_loss):
                    best_val_loss = loggers['val'].get_loss(epoch)
                    best_model = copy.deepcopy(self.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best loss: {:4f}'.format(best_val_loss))

        # load best model weights
        print('Saving model with best validation loss')
        self.load_state_dict(best_model)
        self.save(self.save_path)
        print('done.')


class CobNetLoss(nn.Module):
    def __init__(self, cuda=True):
        super(CobNetLoss, self).__init__()

        self.device = torch.device("cuda:0" if cuda \
                                   else "cpu")

    def forward(self, y_sides, y_fused, target_sides=None, target_fused=None):
        # im: Input image (Tensor)
        # target_sides: Truths for all scales (ResNet part)
        # target_orient: Truths for all orientations (Orientation modules)

        # make transforms to resize target to size of y_sides[s]
        shape_input = y_sides[0].shape[-2:]
        n_elems_inputs = torch.prod(torch.Tensor([shape_input]))
        n_elems_inputs = n_elems_inputs.to(self.device).float()

        loss_sides = torch.Tensor([0]).to(self.device)
        if (target_sides is not None):
            for s in y_sides.keys():

                # Apply transform to batch
                #t_  = [resize_trgt[s](target_sides[s][b, ...])
                #        for b in range(target_sides[s].shape[0])]
                t_ = torch.cat([
                    target_sides[s][b, ...]
                    for b in range(target_sides[s].shape[0])
                ]).float()

                # beta is for class imbalance
                beta = 1 - torch.sum(t_).div(n_elems_inputs)
                #beta = alpha.pow(-1)
                p_plus = utls.clamp_probs(y_sides[s])
                p_neg = utls.clamp_probs(1 - y_sides[s])
                idx_pos = t_ > 0.5
                pos_term = -torch.sum(torch.log(p_plus[np.where(idx_pos)]))

                neg_term = -torch.sum(torch.log(p_neg[np.where(~idx_pos)]))

                loss_sides += beta * pos_term + (1 - beta) * neg_term

        # Compute loss for fused edge map
        loss_fuse = torch.Tensor([0]).to(self.device)
        # make transforms to resize target to size of y_fused
        #resize_transf = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.Resize(y_fused.shape[-2:]),
        #    transforms.ToTensor()])

        if (target_fused is not None):
            # beta is for class imbalance
            #t_  = [resize_transf(target_fused[b, ...])
            #        for b in range(target_fused.shape[0])]
            t_ = torch.cat([
                target_fused[b, ...] for b in range(target_fused.shape[0])
            ]).float()
            beta = 1 - torch.sum(t_).div(n_elems_inputs)
            #beta = alpha.pow(-1)
            p_plus = utls.clamp_probs(y_fused)
            p_neg = utls.clamp_probs(1 - y_fused)
            idx_pos = t_ > 0.5
            pos_term = -torch.sum(torch.log(p_plus[np.where(idx_pos)]))
            neg_term = -torch.sum(torch.log(p_neg[np.where(~idx_pos)]))
            loss_fuse = beta * pos_term + (1 - beta) * neg_term

        #y_orient = dict() # This stores one output per orientation module
        #for orient in self.orientation_keys:
        #    y_orient[orient] = self.forward_on_module(y_scale, im, orient)

        return loss_sides + loss_fuse
