import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import collections
from PIL import Image
from skimage import transform as sk_trfm
import numbers

__author__ = "Wei OUYANG"
__license__ = "GPL"
__version__ = "0.1.0"
__status__ = "Development"


def center_crop(x, center_crop_size):
    assert x.ndim == 3
    centerw, centerh = x.shape[1] // 2, x.shape[2] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[:, centerw - halfw:centerw + halfw, centerh - halfh:centerh +
             halfh]


def to_np_array(x):
    return x.numpy().transpose((1, 2, 0))


def to_tensor(x):
    import torch

    # copy to avoid "negative stride numpy" problem
    x = x.transpose((2, 0, 1)).copy()
    return torch.from_numpy(x).float()


def random_num_generator(config, random_state=np.random):
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    else:
        print(config)
        raise Exception('unsupported format')
    return ret


def poisson_downsampling(image, peak, random_state=np.random):
    if not isinstance(image, np.ndarray):
        imgArr = np.array(image, dtype='float32')
    else:
        imgArr = image.astype('float32')
    Q = imgArr.max(axis=(0, 1)) / peak
    if Q[0] == 0:
        return imgArr
    ima_lambda = imgArr / Q
    noisy_img = random_state.poisson(lam=ima_lambda)
    return noisy_img.astype('float32')


def elastic_transform(image,
                      alpha=1000,
                      sigma=30,
                      spline_order=1,
                      mode='nearest',
                      random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), sigma, mode="constant",
        cval=0) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), sigma, mode="constant",
        cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order,
            mode=mode).reshape(shape)
    return result


class Merge(object):
    """Merge a group of images
    """

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, images):
        if isinstance(images, collections.Sequence) or isinstance(
                images, np.ndarray):
            assert all([isinstance(i, np.ndarray)
                        for i in images]), 'only numpy array is supported'
            shapes = [list(i.shape) for i in images]
            for s in shapes:
                s[self.axis] = None
            assert all([s == shapes[0] for s in shapes
                        ]), 'shapes must be the same except the merge axis'
            return np.concatenate(images, axis=self.axis)
        else:
            raise Exception("obj is not a sequence (list, tuple, etc)")


class Split(object):
    """Split images into individual arraies
    """

    def __init__(self, *slices, **kwargs):
        assert isinstance(slices, collections.Sequence)
        slices_ = []
        for s in slices:
            if isinstance(s, collections.Sequence):
                slices_.append(slice(*s))
            else:
                slices_.append(s)
        assert all([isinstance(s, slice) for s in slices_
                    ]), 'slices must be consist of slice instances'
        self.slices = slices_
        self.axis = kwargs.get('axis', -1)

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            ret = []
            for s in self.slices:
                sl = [slice(None)] * image.ndim
                sl[self.axis] = s
                ret.append(image[sl])
            return ret
        else:
            raise Exception("obj is not an numpy array")


class ElasticTransform(object):
    """Apply elastic transformation on a numpy.ndarray (H x W x C)
    """

    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image):
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma)
        else:
            sigma = self.sigma
        return elastic_transform(image, alpha=alpha, sigma=sigma)


class PoissonSubsampling(object):
    """Poisson subsampling on a numpy.ndarray (H x W x C)
    """

    def __init__(self, peak, random_state=np.random):
        self.peak = peak
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.peak, collections.Sequence):
            peak = random_num_generator(
                self.peak, random_state=self.random_state)
        else:
            peak = self.peak
        return poisson_downsampling(
            image, peak, random_state=self.random_state)


class AddGaussianNoise(object):
    """Add gaussian noise to a numpy.ndarray (H x W x C)
    """

    def __init__(self, mean, sigma, random_state=np.random):
        self.sigma = sigma
        self.mean = mean
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        if isinstance(
                self.mean, collections.Sequence,
                random_state=self.random_state):
            mean = random_num_generator(self.mean)
        else:
            mean = self.mean
        row, col, ch = image.shape
        gauss = self.random_state.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        image += gauss
        return image


class AddSpeckleNoise(object):
    """Add speckle noise to a numpy.ndarray (H x W x C)
    """

    def __init__(self, mean, sigma, random_state=np.random):
        self.sigma = sigma
        self.mean = mean
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        if isinstance(self.mean, collections.Sequence):
            mean = random_num_generator(
                self.mean, random_state=self.random_state)
        else:
            mean = self.mean
        row, col, ch = image.shape
        gauss = self.random_state.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        image += image * gauss
        return image


class GaussianBlurring(object):
    """Apply gaussian blur to a numpy.ndarray (H x W x C)
    """

    def __init__(self, sigma, random_state=np.random):
        self.sigma = sigma
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        image = gaussian_filter(image, sigma=(sigma, sigma, 0))
        return image


class AddGaussianPoissonNoise(object):
    """Add poisson noise with gaussian blurred image to a numpy.ndarray (H x W x C)
    """

    def __init__(self, sigma, peak, random_state=np.random):
        self.sigma = sigma
        self.peak = peak
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        if isinstance(self.peak, collections.Sequence):
            peak = random_num_generator(
                self.peak, random_state=self.random_state)
        else:
            peak = self.peak
        bg = gaussian_filter(image, sigma=(sigma, sigma, 0))
        bg = poisson_downsampling(
            bg, peak=peak, random_state=self.random_state)
        return image + bg


class FloatScaleNumpy(object):
    """scale image from 0.0 to 1.0
    """

    def __init__(self, range_min=0.0, range_max=1.0):
        self.scale = (range_min, range_max)

    def __call__(self, image):
        return image.astype(float) / 255.


class MedianScaleNumpy(object):
    """Scale with median and mean of each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, range_min=0.0, range_max=1.0):
        self.scale = (range_min, range_max)

    def __call__(self, image):
        mn = image.min(axis=(0, 1))
        md = np.median(image, axis=(0, 1))
        return self.scale[0] + (image - mn) * (
            self.scale[1] - self.scale[0]) / (md - mn)


class Normalize(object):
    """Normalize each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image):

        mean_ = self.mean
        std_ = self.std

        if (self.mean is None):
            mean_ = np.mean(image, axis=(0, 1))
        if (self.std is None):
            std_ = np.std(image, axis=(0, 1))

        image -= mean_
        std_[std_ == 0] = 1.0
        image /= std_
        return image


class MutualExclude(object):
    """Remove elements from one channel
    """

    def __init__(self, exclude_channel, from_channel):
        self.from_channel = from_channel
        self.exclude_channel = exclude_channel

    def __call__(self, image):
        mask = image[:, :, self.exclude_channel] > 0
        image[:, :, self.from_channel][mask] = 0
        return image


class ToNumpyArray(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return to_np_array(img)


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return to_tensor(img)


class Void(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img


class RandomCropNumpy(object):
    """Crops the given numpy array at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, random_state=np.random):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.random_state = random_state

    def __call__(self, img):
        w, h = img.shape[:2]
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = self.random_state.randint(0, w - tw)
        y1 = self.random_state.randint(0, h - th)
        return img[x1:x1 + tw, y1:y1 + th, :]


class CenterCropNumpy(object):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[:2]
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[x1:x1 + tw, y1:y1 + th, :]


class RandomFlip(object):
    """Flip a PIL.Image or numpy.ndarray (H x W x C) randomly
    """

    def __init__(self, direction='vertical', random_state=np.random):
        assert isinstance(direction, str)
        self.direction = direction
        if (direction == 'vertical'):
            self.axis = 0
        elif (direction == 'horizontal'):
            self.axis = 1
        else:
            raise Exception('unknown direction')

        self.random_state = random_state

    def __call__(self, image):
        do_it = np.random.choice([True, False])
        if (isinstance(image, np.ndarray)):
            if(do_it):
                image = np.flip(image, axis=self.axis)
        else:
            raise Exception('unsupported type')
        return image


class RandomRotate(object):
    """Rotate a PIL.Image or numpy.ndarray (H x W x C) randomly
    """

    def __init__(self,
                 angle_range=(0.0, 360.0),
                 axes=(0, 1),
                 mode='reflect',
                 random_state=np.random):
        assert isinstance(angle_range, tuple)
        self.angle_range = angle_range
        self.random_state = random_state
        self.axes = axes
        self.mode = mode

    def __call__(self, image):
        angle = self.random_state.uniform(self.angle_range[0],
                                          self.angle_range[1])
        if isinstance(image, np.ndarray):
            mi, ma = image.min(), image.max()
            image = scipy.ndimage.interpolation.rotate(
                image, angle, reshape=False, axes=self.axes, mode=self.mode)
            return np.clip(image, mi, ma)
        elif isinstance(image, Image.Image):
            return image.rotate(angle)
        else:
            raise Exception('unsupported type')


class Resize(object):
    """Resize a PIL.Image or numpy.ndarray (H x W x C)
    """

    def __init__(self, shape, order=3):
        self.output_shape = [shape[0], shape[1]]
        self.order = order

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            return sk_trfm.resize(
                image, self.output_shape, order=self.order, mode='reflect')
        elif isinstance(image, Image.Image):
            return image.resize(self.output_shape, self.order)
        else:
            raise Exception('unsupported type')


class EnhancedCompose(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                assert isinstance(
                    img, collections.Sequence) and len(img) == len(
                        t
                    ), "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected type')
        return img


if __name__ == '__main__':
    from torchvision.transforms import Lambda
    input_channel = 3
    target_channel = 3

    # define a transform pipeline
    transform = EnhancedCompose([
        Merge(),
        RandomCropNumpy(size=(512, 512)),
        RandomRotate(),
        Split([0, input_channel],
              [input_channel, input_channel + target_channel]),
        [CenterCropNumpy(size=(256, 256)),
         CenterCropNumpy(size=(256, 256))],
        [Normalize(), MaxScaleNumpy(0, 1.0)],
        # for non-pytorch usage, remove to_tensor conversion
        [Lambda(to_tensor), Lambda(to_tensor)]
    ])
    # read input data for test
    image_in = np.array(Image.open('input.jpg'))
    image_target = np.array(Image.open('target.jpg'))

    # apply the transform
    x, y = transform([image_in, image_target])
