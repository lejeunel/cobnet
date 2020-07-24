#!/usr/bin/env python3

import glob
import os
import urllib.request
from os.path import join as pjoin

import configargparse
import numpy as np
from models.cobnet import CobNet
import torch
from skimage.io import imsave, imread
from imgaug import augmenters as iaa
from utils.augmenters import Normalize, rescale_augmenter
import matplotlib.pyplot as plt

if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('--model-path', required=True)
    p.add('--in-path', required=True)
    p.add('--out-path', required=True)
    cfg = p.parse_args()

    exts = ['jpg', 'jpeg', 'png']
    if os.path.isdir(cfg.in_path):
        im_paths = []
        for ext in exts:
            im_paths.extend(glob.glob(pjoin(cfg.in_path, '*.' + ext)))
        print('found {} images to process'.format(len(im_paths)))
        if (not os.path.exists(cfg.out_path)):
            os.makedirs(cfg.out_path)
        out_paths = [
            pjoin(cfg.out_path,
                  os.path.split(imp)[-1]) for imp in im_paths
        ]

    elif os.path.isfile(cfg.in_path):
        assert (not os.path.isdir(
            cfg.out)), 'when in is file, give a file name for out'

    model = CobNet()
    state_dict = torch.load(cfg.model_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()

    normalize = iaa.Sequential([
        rescale_augmenter,
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for i, (imp, outp) in enumerate(zip(im_paths, out_paths)):
        print('[{}/{}] {} -> {}'.format(i + 1, len(im_paths), imp, outp))
        im_orig = imread(imp)
        im = normalize(image=im_orig)
        im = torch.from_numpy(np.moveaxis(im, -1, 0)).unsqueeze(0).float()
        res = model(im)
        plt.subplot(131)
        plt.imshow(im_orig)
        plt.subplot(132)
        plt.imshow(res['y_fine'].sigmoid().squeeze().cpu().detach().numpy())
        plt.subplot(133)
        plt.imshow(res['y_coarse'].sigmoid().squeeze().cpu().detach().numpy())
        plt.show()
