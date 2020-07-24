# TODO
- Integrate [Higra](https://github.com/higra/Higra) to generate multiscale hierarchy
- Check training of orientation maps
- Check results with original caffe implementation

# Description
PyTorch implementation of [Convolutional Oriented Boundaries](https://github.com/kmaninis/COB)

## Differences w.r.t original implementation:
- ResNet50 from pyTorch model zoo, which differs from author's Caffe model (has batch normalization layers)
- Batch size of 16
- Base learning-rate is 1e-4 and is increased for "deeper" layers
- Weight initialization is gaussian/normal instead of constant

# Dependencies
Most of these are easily installed with your favorite package manager

- PyTorch >= 1.0
- Numpy
- Scipy
- imgaug
- opencv-python 
- tqdm
- imgaug
- sklearn
- tensorboardX
- higra

# Dataset
Download and uncompress the following datasets/annotations:
- [Pascal VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) 
Unzip this in <root>/pascal-voc
- [Pascal-Context](https://cs.stanford.edu/~roozbeh/pascal-context/) 
Unzip this in <root>/trainval

# Training
On first run, the whole dataset (>10k images) will be processed to extract boundaries, this can take more than an hour!

```sh
python train.py --root-imgs <root>/pascal-voc/VOC2012 --root-segs <root>/trainval --run-path <root>/runs/cob --cuda
```
