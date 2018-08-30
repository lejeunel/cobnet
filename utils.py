from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from skimage import draw
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import convolve
from scipy import io

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

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_image(image_path, transform=None):
    import torch
    import torch.nn as nn
    from torch.autograd import Variable

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """Load an image and convert it to a torch tensor."""

    image = Image.open(image_path)

    if transform:
        image = transform(image)

    #return Variable(image.to(device))
    return image.to(device)

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
