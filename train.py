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
import utils as utls

import params
from cobnet import CobNet
from dataloader import CobDataLoader
from loss import BalancedBCE


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
                for s in res['sides']:
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
        pbar.set_description('[val] lss {:.6e}'.format(loss_))
        pbar.update(1)

    pbar.close()


def train_one_epoch(model, dataloader, optimizers, device, mode, writer,
                    epoch):

    model.train()

    running_loss = 0

    criterion = BalancedBCE()

    pbar = tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):
            for k in optimizers.keys():
                optimizers[k].zero_grad()

            loss_sides = 0
            loss_orient = 0

            if (mode == 'fs'):
                sides = model.forward_sides(data['image'])

                # regress all side-activations
                for s in range(sides.shape[1]):
                    loss_sides += criterion(
                        sides[:, s, ...].unsqueeze(1).sigmoid(), data['cntr'])
                loss_sides.backward()
                optimizers['base'].step()
                optimizers['reduc'].step()

                y_fine, y_coarse = model.forward_fuse(sides.detach())
                loss_fine = criterion(y_fine.sigmoid(), data['cntr'])
                loss_coarse = criterion(y_coarse.sigmoid(), data['cntr'])
                (loss_fine + loss_coarse).backward()
                optimizers['fuse'].step()

                running_loss += loss_sides.cpu().detach().numpy()
                running_loss += loss_coarse.cpu().detach().numpy()
                running_loss += loss_fine.cpu().detach().numpy()

            else:
                res = model(data['image'])
                for i, o_ in enumerate(res['orientations']):
                    loss_orient += criterion(res['orientations'][i],
                                             (data['or_cntr'] == i +
                                              1).float())
                loss_orient.backward()
                optimizers['orientation'].step()
                running_loss += loss_orient.cpu().detach().numpy()

        loss_ = running_loss / ((i + 1) * cfg.batch_size)
        niter = epoch * len(dataloader) + i
        writer.add_scalar('train/loss_{}'.format(mode), loss_, niter)
        pbar.set_description('[train] lss {:.6e}'.format(loss_))
        pbar.update(1)

    pbar.close()

    loss = running_loss / (dataloader.batch_size * len(dataloader))

    out = {'train/loss_{}'.format(mode): loss}

    return out


def train(cfg, model, device, dataloaders, run_path, writer):

    optimizers = {
        'base':
        optim.SGD(model.base_model.parameters(),
                  lr=cfg.lr,
                  weight_decay=cfg.decay,
                  momentum=cfg.momentum),
        'reduc':
        optim.SGD(model.reducers.parameters(),
                  lr=cfg.lr,
                  weight_decay=cfg.decay,
                  momentum=cfg.momentum),
        'fuse':
        optim.SGD(model.fuse.parameters(),
                  lr=cfg.lr,
                  weight_decay=cfg.decay,
                  momentum=cfg.momentum),
        'orientation':
        optim.SGD(model.orientations.parameters(),
                  lr=cfg.lr,
                  weight_decay=cfg.decay,
                  momentum=cfg.momentum),
    }
    lr_sch = {
        'base':
        optim.lr_scheduler.MultiStepLR(optimizers['base'],
                                       milestones=[cfg.epochs_pre]),
        'reduc':
        optim.lr_scheduler.MultiStepLR(optimizers['reduc'],
                                       milestones=[cfg.epochs_pre]),
        'fuse':
        optim.lr_scheduler.MultiStepLR(optimizers['fuse'],
                                       milestones=[cfg.epochs_pre]),
        'orientation':
        optim.lr_scheduler.MultiStepLR(optimizers['orientation'],
                                       milestones=[cfg.epochs_pre])
    }

    mode = 'fs'
    for epoch in range(cfg.epochs):
        if (epoch > cfg.epochs_pre - 1):
            mode = 'or'
        # save checkpoint
        if (epoch % cfg.cp_period == 0):
            path = pjoin(run_path, 'checkpoints', 'cp_{}.pth.tar'.format(mode))
            utls.save_checkpoint({'epoch': epoch + 1, 'model': model}, path)
        # save previews
        if (epoch % cfg.cp_period == 0) and (epoch > 0):
            out_path = pjoin(cfg.run_path, 'previews')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print('generating previews to {}'.format(out_path))

            batch = utls.batch_to_device(next(iter(dataloaders['prev'])),
                                         device)
            model.eval()
            with torch.no_grad():
                res = model(batch['image'])
            utls.save_preview(batch, res,
                              pjoin(out_path, 'ep_{:04d}.png'.format(epoch)))

        print('epoch {}/{}. Mode: {}'.format(epoch + 1, cfg.epochs, mode))
        model.train()
        losses = train_one_epoch(model, dataloaders['train'], optimizers,
                                 device, mode, writer, epoch)
        # write losses to tensorboard
        val(model, dataloaders['train'], device, mode, writer, epoch)

        for k in lr_sch.keys():
            lr_sch[k].step()


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
    dl_train = DataLoader(dset_train,
                          collate_fn=dset_train.collate_fn,
                          batch_size=cfg.batch_size,
                          drop_last=True,
                          shuffle=True)

    dset_val = CobDataLoader(root_imgs=cfg.root_imgs,
                             root_segs=cfg.root_segs,
                             split='val')

    dl_val = DataLoader(dset_val, collate_fn=dset_val.collate_fn)
    dl_prev = DataLoader(dset_val,
                         shuffle=True,
                         batch_size=cfg.n_ims_test,
                         collate_fn=dset_val.collate_fn)

    dataloaders = {'train': dl_train, 'prev': dl_prev, 'val': dl_val}

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
