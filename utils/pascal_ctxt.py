import os
from os.path import join as pjoin
import collections
import json
import numpy as np
from skimage.io import imsave, imread
import scipy.io as io
import matplotlib.pyplot as plt
import glob


class pascalVOCContextLoader:
    """Data loader for the Pascal VOC semantic segmentation dataset.

    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """
    def __init__(self, root_imgs, root_segs, split='train'):
        self.root_imgs = root_imgs
        self.root_segs = root_segs

        self.splits = ['train', 'val', 'test']
        self.split = split

        self.all_base_names_ctxt = [
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(pjoin(self.root_segs, '*.mat'))
        ]

        # read pascal train and validation sets
        with open(pjoin(root_imgs, 'ImageSets', 'Main', 'train.txt')) as f:
            self.pascal_train = f.readlines()
        self.pascal_train = [x.strip() for x in self.pascal_train]
        with open(pjoin(root_imgs, 'ImageSets', 'Main', 'val.txt')) as f:
            self.pascal_val = f.readlines()
        self.pascal_val = [x.strip() for x in self.pascal_val]

        self.base_names = dict()
        self.base_names['train'] = [
            f for f in self.all_base_names_ctxt if f in self.pascal_train
        ]
        self.base_names['valtest'] = [
            f for f in self.all_base_names_ctxt if f in self.pascal_val
        ]

        self.base_names['val'] = self.base_names[
            'valtest'][:len(self.base_names['valtest']) // 2]
        self.base_names['test'] = self.base_names['valtest'][
            len(self.base_names['valtest']) // 2:]

    def __len__(self):
        return len(self.base_names[self.split])

    def __getitem__(self, index):
        base_name = self.base_names[self.split][index]
        im_path = pjoin(self.root_imgs, 'JPEGImages', base_name + '.jpg')
        lbl_path = pjoin(self.root_segs, base_name + '.mat')

        im = imread(im_path)
        data = io.loadmat(lbl_path)
        lbl = data['LabelMap']

        return {'image': im, 'labels': lbl, 'base_name': base_name}


if __name__ == "__main__":

    root_path = '/home/ubelix/lejeune/data'
    dl = pascalVOCContextLoader(root_imgs=pjoin(root_path, 'pascal-voc',
                                                'VOC2012'),
                                root_segs=pjoin(root_path, 'trainval'))

    c = [0, 0]
    r = 1
    npts = 1000
    theta = np.linspace(0, 2 * np.pi, npts)
    x = c[0] + np.cos(theta)
    y = c[1] + np.sin(theta)

    x_interp, y_interp = contours_to_pts(x, y, n_pts=30)
    angles = segments_to_angles(x_interp, y_interp)
    bins = bin_angles(angles)

    plt.plot(x, y)
    for i in range(x_interp.shape[0] - 1):
        plt.plot((x_interp[i], x_interp[i + 1]),
                 (y_interp[i], y_interp[i + 1]),
                 linewidth=4,
                 color=plt.cm.RdYlBu(bins[i] / bins.max()))
    plt.plot(x_interp, y_interp, 'ro')
    plt.grid()
    plt.show()

    im, lbl = dl[0]
    plt.subplot(121)
    plt.imshow(im)
    plt.subplot(122)
    plt.imshow(lbl)
    plt.show()
