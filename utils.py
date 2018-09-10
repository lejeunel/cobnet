from PIL import Image
from collections import namedtuple
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from skimage import draw
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import convolve
from scipy import io
import skimage
from torchvision import transforms
import os

filt_size = 11
filt_straight = np.zeros((filt_size, filt_size))
filt_straight[:, filt_size//2] = 1
filts = []
filts.append(filt_straight)
filts.append(rotate(filts[0], -20, reshape=False))
filts.append(rotate(filts[1], -20, reshape=False))
filts.append(rotate(filts[2], -20, reshape=False))
filts.append(rotate(filts[3], -20, reshape=False))
filts.append(rotate(filts[4], -20, reshape=False))
filts.append(rotate(filts[5], -20, reshape=False))
filts.append(rotate(filts[6], -20, reshape=False))

filts = [f/np.sum(f) for f in filts]

# This follows semantics of numpy.finfo.
_Finfo = namedtuple('_Finfo', ['eps', 'tiny'])
_FINFO = {
    torch.HalfStorage: _Finfo(eps=0.00097656, tiny=6.1035e-05),
    torch.FloatStorage: _Finfo(eps=1.19209e-07, tiny=1.17549e-38),
    torch.DoubleStorage: _Finfo(eps=2.22044604925e-16, tiny=2.22507385851e-308),
    torch.cuda.HalfStorage: _Finfo(eps=0.00097656, tiny=6.1035e-05),
    torch.cuda.FloatStorage: _Finfo(eps=1.19209e-07, tiny=1.17549e-38),
    torch.cuda.DoubleStorage: _Finfo(eps=2.22044604925e-16, tiny=2.22507385851e-308),
}


def _finfo(tensor):
    r"""
    Return floating point info about a `Tensor`:
    - `.eps` is the smallest number that can be added to 1 without being lost.
    - `.tiny` is the smallest positive number greater than zero
      (much smaller than `.eps`).
    Args:
        tensor (Tensor): tensor of floating point data.
    Returns:
        _Finfo: a `namedtuple` with fields `.eps` and `.tiny`.
    """
    return _FINFO[tensor.storage_type()]

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_image(image_path, transform=None):
    import torch
    import torch.nn as nn
    from torch.autograd import Variable

    """Load an image and convert it to a torch tensor."""

    image = Image.open(image_path)

    if transform:
        image = transform(image)

    return image

def crop_and_concat(self, upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)

def generate_oriented_boundaries(boundary_map):
    # Generate 8 oriented boundary map from single boundary map

    return [convolve(boundary_map.astype(float), f) for f in  filts]

def load_boundaries_bsds(path):
    """
    Load the ground truth boundaries from the Matlab file
    at the specified path.
    :param path: path
    :return: a list of (H,W) arrays, each of which contains a
    boundary ground truth
    """

    gt = io.loadmat(path)
    gts = [gt['groundTruth'][0,i]['Boundaries'][0,0]
           for i in range(gt['groundTruth'][0,:].size)]

    # Return random segmentation for 5 users
    return gts

def save_tensors(im, pred, trgt, path):
    
    path_ = os.path.split(path)[0]
    if(not os.path.exists(path_)):
        os.makedirs(path_)

    #im = (im*255).astype('uint8')

    pred_ = pred.detach().cpu().numpy().transpose((1,2,0))
    pred_ = (np.repeat(pred_, 3, axis=-1)*255).astype('uint8')

    trgt_ = trgt.detach().cpu().numpy().transpose((1,2,0))
    trgt_ = (np.repeat(trgt_, 3, axis=-1)*255).astype('uint8')

    all = np.concatenate((im, pred_, trgt_), axis=1)
    skimage.io.imsave(path, all)

def clamp_probs(probs):
    eps = _finfo(probs).eps
    return probs.clamp(min=eps, max=1 - eps)
