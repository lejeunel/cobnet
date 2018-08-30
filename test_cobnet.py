import torch
import torchvision
from torchvision import transforms
from torch import nn
import os
import glob
import utils as utls
import requests
import pickle as pk
import numpy as np
from skimage import io
from cobnet import CobNet


model_path = os.path.join('models', 'resnet50.pth')
root_dir = os.path.join('/home',
                        'laurent.lejeune',
                        'data',
                        'BSR',
                        'BSDS500',
                        'data')

image_paths = sorted(glob.glob(os.path.join(root_dir, 'images','*', '*.jpg')))
truth_paths = sorted(glob.glob(os.path.join(root_dir, 'groundTruth','*', '*.mat')))

# Image preprocessing
# Trained on ImageNet where images are normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
# We use the same normalization statistics here.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))])


#test = utls.load_boundaries_bsds(truth_paths[0])

train_val_ratio = .9
n_train_smpls = int(len(image_paths)*train_val_ratio)
train_idx = (0, n_train_smpls)
val_idx = (n_train_smpls+1, len(image_paths))

images_train = image_paths[train_idx[0]:train_idx[1]]
truths_train = truth_paths[train_idx[0]:train_idx[1]]

images_val = image_paths[val_idx[0]:val_idx[1]]
truths_val = truth_paths[val_idx[0]:val_idx[1]]

mycobnet = CobNet(img_paths_train=images_train,
                  truth_paths_train=truths_train,
                  img_paths_val=images_val,
                  truth_paths_val=truths_val,
                  transform=transform)

mycobnet.train()

#io.imsave('test.png', pred_)
