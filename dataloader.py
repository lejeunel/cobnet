from torch.utils.data import Dataset, DataLoader
import utils as utls
import numpy
from PIL import Image
from torch.autograd import Variable
import torch
import numpy as np

class CobDataLoader(Dataset):

    def __init__(self,
                 img_paths,
                 truth_paths=None,
                 transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.img_paths = img_paths
        self.truth_paths = truth_paths
        self.transform = transform

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = utls.load_image(img_path,
                                transform=self.transform)

        if(self.truth_paths is not None):
            truth_path = self.truth_paths[idx]
            gts = utls.load_boundaries_bsds(truth_path)

            # Pick segmentation of a random user...
            gt = gts[np.random.choice(len(gts))]
            gt = Image.fromarray(np.uint8(gt*255))

            if self.transform:
                gt = self.transform(gt)

            gt = gt.to(self.device)

        return img, gt
