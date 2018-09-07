from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import utils as utls
import numpy
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import numpy as np

class CobDataLoader(Dataset):

    def __init__(self,
                 img_paths,
                 shapes_of_sides,
                 shape_of_base,
                 mean_norm_base,
                 std_norm_base,
                 truth_paths=None):
        """
        Args:
        """

        self.img_paths = img_paths
        self.truth_paths = truth_paths

        self.shapes_of_sides = shapes_of_sides

        self.im_transform = transforms.Compose([
            transforms.Resize(shape_of_base),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm_base,
                                 std=std_norm_base)])

        mean_inv_normalize = -np.asarray(mean_norm_base)/np.asarray(std_norm_base)
        std_inv_normalize = 1./np.asarray(std_norm_base)
        self.im_inv_transform = transforms.Compose([
            transforms.Normalize(mean=mean_inv_normalize,
                                 std=std_inv_normalize),
            transforms.ToPILImage(),
            transforms.Resize(np.asarray(shape_of_base)//2)])

        self.truth_transform_fuse = transforms.Compose([
            transforms.Resize(np.asarray(shape_of_base)//2),
            transforms.ToTensor()])

        self.truth_transforms_sides = [transforms.Compose([
            transforms.Resize((w, h)),
            transforms.ToTensor()]) for w,h in self.shapes_of_sides]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # When truths are resized, the values change
        # we apply a threshold to get back to binary

        thr = torch.Tensor([0.]).to(self.device)

        img = utls.load_image(img_path, self.im_transform).to(self.device)
        img.requires_grad = True

        if(self.truth_paths is not None):
            truth_path = self.truth_paths[idx]
            gts = utls.load_boundaries_bsds(truth_path)

            # Pick segmentation of a random user...
            gt = gts[np.random.choice(len(gts))]
            gt = Image.fromarray(np.uint8(gt*255))

            gts_sides = [trf(gt).to(self.device) > thr
                         for trf in self.truth_transforms_sides]

            gt_fuse = self.truth_transform_fuse(gt).to(self.device) > thr

        return img, gts_sides, gt_fuse
