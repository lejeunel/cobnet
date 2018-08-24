import os
import utils as utls
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
from bsds_dataset import BSDSDataset
import utils as utls

bsds = BSDSDataset(os.path.join('/home',
                                'laurent.lejeune',
                                'data',
                                'BSR'))


ind = 0
name = bsds.train_sample_names[ind]

bnd = bsds.boundaries(name)[0]
img = bsds.read_image(name)

or_bnds = utls.generate_oriented_boundaries(bnd)

plt.subplot(331);plt.imshow(img);
plt.subplot(332);plt.imshow(bnd);
plt.subplot(333);plt.imshow(or_bnds[0]);
plt.subplot(334);plt.imshow(or_bnds[1]);
plt.subplot(335);plt.imshow(or_bnds[2]);
plt.subplot(336);plt.imshow(or_bnds[3]);
plt.subplot(337);plt.imshow(or_bnds[4]);
plt.subplot(338);plt.imshow(or_bnds[5]);
plt.subplot(339);plt.imshow(or_bnds[6]);
plt.tight_layout()
plt.show()
#plt.subplot(121); plt.imshow(gt_im[..., 0])
#plt.subplot(122); plt.imshow(gt_im[..., 1])
#plt.show()
