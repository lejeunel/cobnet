from torchvision import transforms as trfms
from torchvision.transforms import Lambda
from torch.utils.data import Dataset, DataLoader
import utils as utls
import im_utils as im_utls
import numpy
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import numpy as np
from skimage.morphology import disk
from skimage.morphology import binary_dilation
import collections

class CobDataLoader(Dataset):

    def __init__(self,
                 img_paths,
                 shapes_of_sides,
                 shape_of_base,
                 mean_norm_base,
                 std_norm_base,
                 truth_paths=None,
                 phase = 'train'):
        """
        Args:
        """

        self.img_paths = img_paths
        self.truth_paths = truth_paths

        self.shapes_of_sides = shapes_of_sides
        self.phase = phase

        self.shape_of_base = shape_of_base

        self.trfm_normalize = trfms.Compose([
            trfms.Resize(shape_of_base),
            trfms.ToTensor(),
            trfms.Normalize(mean=mean_norm_base,
                                 std=std_norm_base)])

        mean_inv_normalize = -np.asarray(mean_norm_base)/np.asarray(std_norm_base)
        std_inv_normalize = 1./np.asarray(std_norm_base)

        self.trfm_in_normalize = im_utls.EnhancedCompose([
            im_utls.FloatScaleNumpy(),
            im_utls.Normalize(mean=mean_inv_normalize,
                              std=std_inv_normalize)])

        self.trfm_half_in_shape = trfms.Compose([
            trfms.ToPILImage(),
            trfms.Resize(
                np.asarray(shape_of_base)//2),
            trfms.ToTensor()])

        self.trfm_in_shape = trfms.Compose([
            trfms.Resize(
                np.asarray(shape_of_base)),
            trfms.ToTensor()])

        self.trfm_sides = [im_utls.EnhancedCompose([
            im_utls.ToNumpyArray(),
            im_utls.Resize((w, h)),
            im_utls.ToTensor()]) for w,h in self.shapes_of_sides]

        # Comes as im, labels
        self.trfm_augment = im_utls.EnhancedCompose([
            [im_utls.Resize(self.shape_of_base)]*2,
            im_utls.Merge(),
            im_utls.RandomRotate(),
            im_utls.RandomFlip(direction='horizontal'),
            im_utls.RandomFlip(direction='vertical'),
            im_utls.Split([0, 3], [3, 4]),
            [im_utls.Normalize(mean=mean_norm_base, std=std_norm_base),
             im_utls.Void()],
            [Lambda(im_utls.to_tensor)]*2
        ])

        # We dilate segmentations
        self.selem = disk(3)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # When truths are resized, the values change
        # we apply a threshold to get back to binary

        thr = torch.Tensor([0.])

        img = np.asarray(utls.load_image(img_path, self.trfm_in_shape))
        img = (img*255).astype(np.uint8).transpose((1,2,0))
        #img.requires_grad = True

        if(self.truth_paths is not None): # we're in train mode
            truth_path = self.truth_paths[idx]
            gts = utls.load_boundaries_bsds(truth_path)
            gts = [binary_dilation(np.asarray(g), selem=self.selem)
                   for g in gts]
             
            gt = ((np.sum(gts,axis=0) > 3)*255).astype(np.uint8)[..., np.newaxis]

            # Apply data augmentation
            im, gt = self.trfm_augment([img, gt])

            img = self.trfm_in_normalize(img)

            import pdb; pdb.set_trace()
            gts_sides = [trf(gt) > thr
                         for trf in self.trfm_sides]

            gt_fuse = self.trfm_half_in_shape(gt) > thr

        # Move to device
        img = img.to(self.device)
        gt_fuse = gt_fuse.to(self.device)
        gts_sides = [g.to(self.device) for g in gts_sides]

        return img, gts_sides, gt_fuse
