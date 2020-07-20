import math
import os
from os.path import join as pjoin

import cv2
import imgaug as ia
import matplotlib.pyplot as plt
import numpy as np
import torch
from imgaug import augmenters as iaa
from scipy import sparse
from scipy.interpolate import interp1d
from sklearn.metrics import pairwise_distances_argmin_min
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.augmenters import Normalize, rescale_augmenter
from utils.pascal_ctxt import pascalVOCContextLoader


def interpolate_to_polygon(arr, n_pts=10000, n_bins=8):
    # arr is an integer array
    contours = np.zeros(arr.shape)
    for c in np.unique(arr):
        arr_ = arr == c
        # label = measure.label(arr_)
        # for l in np.unique(label):
        #     if (l > 0):
        contours_, _ = cv2.findContours(
            (arr_).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours_ = np.squeeze(contours_[np.argmax([len(c)
        #                                             for c in contours_])])

        for cntr in contours_:
            if (cntr.shape[0] > 3):
                pts_contour = cntr.squeeze()
                # switch to x,y reference
                x = pts_contour[:, 0]
                y = arr.shape[0] - pts_contour[:, 1]
                # y = pts_contour[:, 1]
                bins = bin_contour(x, y, n_bins=n_bins, n_pts=n_pts)
                i, j = arr.shape[0] - y, x
                i = np.clip(i, 0, arr.shape[0] - 1)
                j = np.clip(j, 0, arr.shape[1] - 1)
                contours[i, j] = bins + 1

    # remove edges at borders
    contours[0, :] = 0
    contours[-1, :] = 0
    contours[:, 0] = 0
    contours[:, -1] = 0

    return contours


def contours_to_pts(x, y, n_pts=100):
    pts = np.concatenate((x[..., None], y[..., None]), axis=1)
    distance = np.cumsum(np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    interpolator = interp1d(distance, pts, kind='linear', axis=0)
    alpha = np.linspace(0, 1, n_pts)

    interp_pts = interpolator(alpha)
    interp_pts = np.concatenate((interp_pts, interp_pts[0, :][None, ...]))

    return interp_pts[:, 0], interp_pts[:, 1]


def segments_to_angles(x, y):
    pts = np.concatenate((x[..., None], y[..., None]), axis=1)
    dx = pts[:-1, 0] - pts[1:, 0]
    dy = pts[:-1, 1] - pts[1:, 1]
    tan = dy / (dx + 1e-8)
    angles = np.arctan(tan)
    # angles = np.unwrap(angles, np.pi / 2)
    # angles[angles < 0] = angles[angles < 0] + np.pi / 2
    # angles = np.arctan2(dy, dx)

    return angles


def bin_contour(x, y, n_bins=8, n_pts=10000):
    pts = np.concatenate((x[..., None], y[..., None]), axis=1)
    x_interp, y_interp = contours_to_pts(x, y, n_pts=n_pts)

    # calculate mid-points of each segment
    pts_interp = np.concatenate((x_interp[..., None], y_interp[..., None]),
                                axis=1)
    vec = pts_interp[1:, :] - pts_interp[:-1, :]
    M = pts_interp[:-1, :] + vec

    seg_idx, _ = pairwise_distances_argmin_min(pts, M)

    angles = segments_to_angles(x_interp, y_interp)
    inds = bin_angles(angles, n_bins=8)

    bins = inds[seg_idx]
    return bins


def bin_angles(angles, n_bins=8):

    # shift to [0, pi]
    angles += np.pi / 2
    angles -= np.pi / n_bins / 2

    bins = np.linspace(0, np.pi - np.pi / n_bins, n_bins)

    inds = np.digitize(angles, bins)
    inds[inds == n_bins] = 0
    return inds


class CobDataLoader(Dataset):
    def __init__(self,
                 root_imgs,
                 root_segs,
                 augmentations=None,
                 normalization_mean=[0.485, 0.456, 0.406],
                 normalization_std=[0.229, 0.224, 0.225],
                 resize_shape=512,
                 split='train'):
        """
        Args:
        """

        self.root_segs = root_segs
        self.root_imgs = root_imgs
        self.dl = pascalVOCContextLoader(root_imgs, root_segs, split=split)

        self.reshaper = iaa.Noop()
        self.augmentations = iaa.Noop()

        if (augmentations is not None):
            self.augmentations = augmentations

        if (resize_shape is not None):
            self.reshaper = iaa.size.Resize(resize_shape)

        self.normalization = iaa.Sequential([
            rescale_augmenter,
            Normalize(mean=normalization_mean, std=normalization_std)
        ])

        self.or_cntr_path = pjoin(
            os.path.split(self.root_segs)[0], 'orientated_contours')

        self.prepare_all()

    def prepare_all(self):
        if (not os.path.exists(self.or_cntr_path)):
            os.makedirs(self.or_cntr_path)
            print('preparing orientation maps to {}'.format(self.or_cntr_path))

            dl = pascalVOCContextLoader(self.root_imgs, self.root_segs)
            for s in dl.splits:
                dl.split = s
                for ii in tqdm(range(len(dl))):
                    s = dl[ii]
                    or_cntr = interpolate_to_polygon(s['labels']).astype(
                        np.uint8)
                    or_cntr = sparse.csr_matrix(or_cntr)
                    sparse.save_npz(
                        pjoin(self.or_cntr_path, s['fname'] + '.npz'), or_cntr)

    def __len__(self):
        return len(self.dl)

    def __getitem__(self, idx):
        sample = self.dl[idx]

        or_cntr = sparse.load_npz(
            pjoin(self.or_cntr_path, sample['base_name'] + '.npz'))
        sample['or_cntr'] = or_cntr.toarray()

        aug = iaa.Sequential(
            [self.reshaper, self.augmentations, self.normalization])
        aug_det = aug.to_deterministic()

        sample['or_cntr'] = ia.SegmentationMapsOnImage(
            sample['or_cntr'], shape=sample['or_cntr'].shape)

        sample['image'] = aug_det(image=sample['image'])
        sample['or_cntr'] = aug_det(
            segmentation_maps=sample['or_cntr']).get_arr()[..., None]
        sample['cntr'] = sample['or_cntr'].astype(bool)

        return sample

    @staticmethod
    def collate_fn(data):

        to_collate = ['image', 'or_cntr', 'cntr']

        out = dict()
        for k in data[0].keys():
            if (k in to_collate):
                out[k] = torch.stack([
                    torch.from_numpy(np.rollaxis(data[i][k], -1)).float()
                    for i in range(len(data))
                ])
            else:
                out[k] = [data[i][k] for i in range(len(data))]

        return out


if __name__ == "__main__":
    # c = [0, 0]
    # r = 1
    # npts = 1000
    # theta = np.linspace(0, 2 * np.pi, npts)
    # x = c[0] + np.cos(theta)
    # y = c[1] + np.sin(theta)

    # bins = bin_contour(x, y)

    # for i in range(x.shape[0]):
    #     plt.plot(x[i], y[i], 'o', color=plt.cm.RdYlBu(bins[i] / 8))

    # plt.grid()
    # plt.show()

    transf = iaa.Sequential([
        iaa.Flipud(p=0.5),
        iaa.Fliplr(p=.5),
        iaa.Fliplr(p=.5),
        iaa.Rotate([11.25 * i for i in range(16)])
    ])
    root_path = '/home/ubelix/lejeune/data'
    dl = CobDataLoader(root_imgs=pjoin(root_path, 'pascal-voc', 'VOC2012'),
                       root_segs=pjoin(root_path, 'trainval'),
                       augmentations=transf)
    dl = DataLoader(dl, collate_fn=dl.collate_fn)

    for d in dl:
        or_cntr = d['or_cntr'].detach().cpu().numpy().squeeze()
        im = d['image'].detach().cpu().numpy().squeeze()

        plt.subplot(121)
        plt.imshow(im)
        plt.subplot(122)
        plt.imshow(or_cntr)
        plt.show()
