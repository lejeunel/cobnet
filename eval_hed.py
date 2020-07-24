#!/usr/bin/env python3

import glob
import os
import urllib.request
from os.path import join as pjoin

import configargparse
import cv2 as cv
import numpy as np

model_weights_url = 'http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel'
model_weights_file = 'hed_pretrained_bsds.caffemodel'

model_arch_file = 'hed_pretrained_bsds.prototxt'
model_arch_url = 'https://raw.githubusercontent.com/legolas123/cv-tricks.com/master/OpenCV/Edge_detection/deploy.prototxt'


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


def get_model(model_root_path=os.path.expanduser(pjoin('~', '.models'))):

    if not os.path.exists(model_root_path):
        os.makedirs(model_root_path)

    model_weights_path = os.path.expanduser(
        pjoin(model_root_path, model_weights_file))

    model_arch_path = os.path.expanduser(
        pjoin(model_root_path, model_arch_file))

    if (not os.path.exists(model_weights_path)):
        print('Downloading HED model weights to {}'.format(model_weights_path))
        urllib.request.urlretrieve(model_weights_url, model_weights_path)

    if (not os.path.exists(model_arch_path)):
        print('Downloading HED model prototype to {}'.format(model_arch_path))
        urllib.request.urlretrieve(model_arch_url, model_arch_path)

    model = cv.dnn.readNet(model_arch_path, model_weights_path)
    cv.dnn_registerLayer('Crop', CropLayer)

    return model


def do_pb_single(im_path, model):

    im = cv.imread(im_path)
    inp = cv.dnn.blobFromImage(im,
                               scalefactor=1.0,
                               size=(512, 512),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False,
                               crop=False)
    model.setInput(inp)
    out = model.forward()
    out = out[0, 0]
    out = cv.resize(out, (im.shape[1], im.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)

    return out


if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('--in-path', required=True)
    p.add('--out-path', required=True)
    cfg = p.parse_args()

    exts = ['jpg', 'jpeg', 'png']
    if os.path.isdir(cfg.in_path):
        im_paths = []
        for ext in exts:
            im_paths.extend(glob.glob(pjoin(cfg.in_path, '*.' + ext)))
        print('found {} images to process'.format(len(im_paths)))
        if (not os.path.exists(cfg.out_path)):
            os.makedirs(cfg.out_path)
        out_paths = [
            pjoin(cfg.out_path,
                  os.path.split(imp)[-1]) for imp in im_paths
        ]

    elif os.path.isfile(cfg.in_path):
        assert (not os.path.isdir(
            cfg.out)), 'when in is file, give a file name for out'

    model = get_model()
    for i, (imp, outp) in enumerate(zip(im_paths, out_paths)):
        print('[{}/{}] {} -> {}'.format(i + 1, len(im_paths), imp, outp))
        if (not os.path.exists(outp)):
            out = do_pb_single(imp, model)
            cv.imwrite(outp, out)
        else:
            print('{} already exists.'.format(outp))
