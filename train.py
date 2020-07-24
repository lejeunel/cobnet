#!/usr/bin/env python3

import os
from os.path import join as pjoin

import numpy as np
import torch
import torch.optim as optim
import yaml
from imgaug import augmenters as iaa
from skimage import io
from tensorboardX import SummaryWriter
from torch import sigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import utils as utls
import params
from models.cobnet import CobNet
from utils.dataloader import CobDataLoader
from utils.loss import BalancedBCE

import math


def freeze_bn(m):
    # we update the running stats and freeze gamma/beta only
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # m.eval()
        # m.weight.requires_grad = False
        # m.bias.requires_grad = False
        pass


def make_data_aug(cfg):
    transf = iaa.Sequential([
        iaa.Flipud(p=0.5),
        iaa.Fliplr(p=.5),
        iaa.Fliplr(p=.5),
        iaa.Rotate(
            [360 / cfg.aug_n_angles * i for i in range(cfg.aug_n_angles)])
    ])

    return transf


def val(model, dataloader, device, mode, writer, epoch):

    model.eval()

    running_loss = 0

    criterion = BalancedBCE()

    pbar = tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.no_grad():

            res = model(data['image'])

            loss_sides = 0
            loss_orient = 0

            if (mode == 'fs'):
                for s in res['late_sides']:
                    loss_sides += criterion(s.sigmoid(), data['cntr'])

                loss_fine = criterion(res['y_fine'].sigmoid(), data['cntr'])
                loss_coarse = criterion(res['y_coarse'].sigmoid(),
                                        data['cntr'])
                running_loss += loss_sides.cpu().detach().numpy()
                running_loss += loss_coarse.cpu().detach().numpy()
                running_loss += loss_fine.cpu().detach().numpy()

            else:
                for i, o_ in enumerate(res['orientations']):
                    loss_orient += criterion(res['orientations'][i],
                                             (data['or_cntr'] == i +
                                              1).float())
                running_loss += loss_orient.cpu().detach().numpy()

        loss_ = running_loss / ((i + 1) * cfg.batch_size)
        niter = epoch * len(dataloader) + i
        writer.add_scalar('val/loss_{}'.format(mode), loss_, niter)
        pbar.set_description('[val] lss {:.3e}'.format(loss_))
        pbar.update(1)

    pbar.close()


def check_nan_inf(tnsr):
    n_nan = torch.isnan(tnsr).sum()
    n_inf = torch.isinf(tnsr).sum()

    return n_nan + n_inf


def train_one_epoch(model, dataloaders, optimizers, device, mode, writer,
                    epoch):

    running_loss = 0

    criterion = BalancedBCE()

    if (mode == 'fs'):
        dataloader = dataloaders['train_fs']
    else:
        dataloader = dataloaders['train_or']

    pbar = tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):

            if (mode == 'fs'):

                loss = 0
                _, sides = model.forward_sides(data['image'])

                # regress all side-activations
                for s in sides:
                    loss += criterion(s, data['cntr'])

                # utls.print_grad_norms(model)

                y_fine, y_coarse = model.forward_fuse(sides)

                loss += criterion(y_fine, data['cntr'])
                loss += criterion(y_coarse, data['cntr'])
                loss.backward()

                optimizers['base'].step()
                optimizers['reduc'].step()
                optimizers['fuse'].step()

                optimizers['base'].zero_grad()
                optimizers['reduc'].zero_grad()
                optimizers['fuse'].zero_grad()

                running_loss += loss.cpu().detach().numpy()

            else:
                loss_orient = 0
                optimizers['orientation'].zero_grad()
                res = model(data['image'])
                for j, o_ in enumerate(res['orientations']):
                    loss_orient += criterion(res['orientations'][j],
                                             (data['or_cntr'] == j +
                                              1).float())
                loss_orient.backward()
                optimizers['orientation'].step()
                running_loss += loss_orient.cpu().detach().numpy()

        loss_ = running_loss / ((i + 1) * dataloader.batch_size)
        niter = epoch * len(dataloader) + i
        writer.add_scalar('train/loss_{}'.format(mode), loss_, niter)
        pbar.set_description('[train] lss {:.3e}'.format(loss_))
        pbar.update(1)

    pbar.close()

    loss = running_loss / (dataloader.batch_size * len(dataloader))

    out = {'train/loss_{}'.format(mode): loss}

    return out


def train(cfg, model, device, dataloaders, run_path, writer):

    model_params = utls.parse_model_params(model)

    optimizers = {
        'base':
        optim.SGD([
            {
                'params': model_params['base0-3.weight'],
                'lr': cfg.lr
            },
            {
                'params': model_params['base0-3.bias'],
                'weight_decay': 0,
                'lr': cfg.lr * 2
            },
            {
                'params': model_params['base4.weight'],
                'lr': cfg.lr * 100
            },
            {
                'params': model_params['base4.bias'],
                'weight_decay': 0,
                'lr': cfg.lr * 200
            },
        ],
                  lr=cfg.lr,
                  weight_decay=cfg.decay,
                  momentum=cfg.momentum),
        'reduc':
        optim.SGD([{
            'params': model_params['reducers.weight'],
            'lr': cfg.lr * 100,
        }, {
            'params': model_params['reducers.bias'],
            'lr': cfg.lr * 200,
            'weight_decay': 0,
        }],
                  weight_decay=cfg.decay,
                  momentum=cfg.momentum),
        'fuse':
        optim.SGD([{
            'params': model_params['fuse.weight'],
            'lr': cfg.lr * 100,
        }, {
            'params': model_params['fuse.bias'],
            'lr': cfg.lr * 200,
            'weight_decay': 0,
        }],
                  weight_decay=cfg.decay,
                  momentum=cfg.momentum),
        'orientation':
        optim.SGD([{
            'params': model_params['orientation.weight'],
            'lr': cfg.lr
        }, {
            'params': model_params['orientation.bias'],
            'lr': cfg.lr * 2,
            'weight_decay': 0,
        }],
                  weight_decay=cfg.decay,
                  momentum=cfg.momentum),
    }
    lr_sch = {
        'base':
        optim.lr_scheduler.MultiStepLR(optimizers['base'],
                                       milestones=[cfg.epochs_div_lr],
                                       gamma=0.1),
        'reduc':
        optim.lr_scheduler.MultiStepLR(optimizers['reduc'],
                                       milestones=[cfg.epochs_div_lr],
                                       gamma=0.1),
        'fuse':
        optim.lr_scheduler.MultiStepLR(optimizers['fuse'],
                                       milestones=[cfg.epochs_div_lr],
                                       gamma=0.1)
    }

    fs_path = pjoin(run_path, 'checkpoints', 'cp_fs.pth.tar')
    if (os.path.exists(fs_path)):
        print('found model {}'.format(fs_path))
        print('will skip fusion mode')
        start_epoch = cfg.epochs_pre
    else:
        start_epoch = 0
        mode = 'fs'

    for epoch in range(start_epoch, cfg.epochs):
        if (epoch > cfg.epochs_pre - 1):
            mode = 'or'

        print('epoch {}/{}, mode: {}, lr: {:.2e}'.format(
            epoch + 1, cfg.epochs, mode, lr_sch['base'].get_last_lr()[0]))
        writer.add_scalar('base_lr', lr_sch['base'].get_last_lr()[0], epoch)
        model.train()
        model.base_model.apply(freeze_bn)

        losses = train_one_epoch(model, dataloaders, optimizers, device, mode,
                                 writer, epoch)

        # save checkpoint
        path = pjoin(run_path, 'checkpoints', 'cp_{}.pth.tar'.format(mode))
        utls.save_checkpoint({'epoch': epoch + 1, 'model': model}, path)

        # save previews
        out_path = pjoin(cfg.run_path, 'previews')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print('generating previews to {}'.format(out_path))

        batch = utls.batch_to_device(next(iter(dataloaders['prev'])), device)
        model.eval()
        with torch.no_grad():
            res = model(batch['image'])
        utls.save_preview(batch, res,
                          pjoin(out_path, 'ep_{:04d}.png'.format(epoch)))
        for k in lr_sch.keys():
            lr_sch[k].step()

        # write losses to tensorboard
        model.eval()
        val(model, dataloaders['train'], device, mode, writer, epoch)


def main(cfg):

    if (not os.path.exists(cfg.run_path)):
        os.makedirs(cfg.run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model = CobNet()
    model.to(device)

    transf = make_data_aug(cfg)

    dset_train = CobDataLoader(root_imgs=cfg.root_imgs,
                               root_segs=cfg.root_segs,
                               augmentations=transf,
                               split='train')
    dl_train_fs = DataLoader(dset_train,
                             collate_fn=dset_train.collate_fn,
                             batch_size=cfg.batch_size,
                             drop_last=True,
                             shuffle=True)
    dl_train_or = DataLoader(dset_train,
                             collate_fn=dset_train.collate_fn,
                             batch_size=4,
                             drop_last=True,
                             shuffle=True)

    dset_val = CobDataLoader(root_imgs=cfg.root_imgs,
                             root_segs=cfg.root_segs,
                             split='val')

    dl_val = DataLoader(dset_val,
                        collate_fn=dset_val.collate_fn,
                        batch_size=32)
    dl_prev = DataLoader(dset_val,
                         batch_size=cfg.n_ims_test,
                         collate_fn=dset_val.collate_fn)

    dataloaders = {
        'train_fs': dl_train_fs,
        'train_or': dl_train_or,
        'prev': dl_prev,
        'val': dl_val
    }

    # Save cfg
    with open(pjoin(cfg.run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    writer = SummaryWriter(cfg.run_path, flush_secs=1)
    train(cfg, model, device, dataloaders, cfg.run_path, writer)

    return model


if __name__ == "__main__":

    p = params.get_params()

    p.add('--root-imgs', required=True)
    p.add('--root-segs', required=True)
    p.add('--run-path', required=True)

    cfg = p.parse_args()

    main(cfg)
