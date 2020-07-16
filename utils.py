import torch
import os
from torchvision.utils import make_grid, save_image
import numpy as np


def batch_to_device(batch, device):

    return {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }


def save_checkpoint(dict_, path):

    dir_ = os.path.split(path)[0]
    if (not os.path.exists(dir_)):
        os.makedirs(dir_)

    state_dict = dict_['model'].state_dict()

    torch.save(state_dict, path)


def save_preview(data, res, path, n_orient=4):
    ims = make_grid(data['image'], scale_each=True, nrow=1)
    fine = make_grid(res['y_fine'].sigmoid(), nrow=1)
    coarse = make_grid(res['y_coarse'].sigmoid(), nrow=1)

    total_orients = len(res['orientations'])
    orients_idx = np.arange(0, total_orients, step=total_orients // n_orient)

    orients = []
    for i in orients_idx:

        o_ = make_grid(res['orientations'][i].sigmoid(), nrow=1)
        orients.append(o_)

    save_image([ims, fine, coarse, *orients], path)
