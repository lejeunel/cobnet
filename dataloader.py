from torch.utils.data import Dataset, DataLoader
import utils as utls
import numpy
from PIL import Image
from torch.autograd import Variable
import torch

class CobDataLoader(Dataset):

    def __init__(self,
                 img_paths,
                 truth_paths=None,
                 transform=None,
                 batch_size=4,
                 shuffle=True,
                 num_workers=4):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.img_paths = img_paths
        self.truth_paths = truth_paths
        self.transform = transform

        # Make iterator
        self.iter_ = DataLoader(self)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = utls.load_image(img_path,
                                transform=self.transform)

        if(self.truth_paths is not None):
            truth_path = self.truth_paths[idx]
            gts = utls.load_boundaries(truth_path)

            # Pick segmentation of a random user...
            gt = gts[np.random.choice(len(gts))]
            gt = Image.fromarray(np.uint8(gt*255))

            if transform:
                gt = self.transform(gt).unsqueeze(0)

            gt = Variable(gt.to(self.device))

        return img, 
