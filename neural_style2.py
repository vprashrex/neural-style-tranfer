import os
import math
import re
from argparse import ArgumentParser
from collections import OrderedDict

from PIL import Image
import numpy as np
import scipy.misc

from stylize import stylize
import imageio
import cv2
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import argparse

# default arguments
progress_write = True
progress_plot = True
CONTENT_WEIGHT = 5e0
#CONTENT_WEIGHT = 7.5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
#STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
#TV_WEIGHT = 2e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
#LEARNING_RATE = 1e0
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 10
network = '../imagenet-vgg-verydeep-19.mat'
POOLING = 'max'
content_img = '1.jpg'
style_img = 'style.jpg'
widths = 512
output = 'result.jpg'
checkpoint_output = 'output_{:05}.jpg'

def build_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--content',dest='content',help='content image',metavar='CONTENT',required=True)
    parser.add_argument('--styles',dest='styles',help='one or more style images',metavar='STYLE',required=True)
    parser.add_argument('--network',dest='network',help='path to network parameters',metavar='VGG_PATH',required=True)
    
    return parser
    

def fmt_imsave(fmt, iteration):
    if re.match(r'^.*\{.*\}.*$', fmt):
        return fmt.format(iteration)
    elif '%' in fmt:
        return fmt % iteration
    else:
        raise ValueError("illegal format string '{}'".format(fmt))


def main():
    parser = build_parser()
    options = parser.parse_args()
    # https://stackoverflow.com/a/42121886
    key = 'TF_CPP_MIN_LOG_LEVEL'
    if key not in os.environ:
        os.environ[key] = '2'

    content_image = imread(options.content)

    style_image = imread(options.styles)

    width = widths
    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                                    content_image.shape[1] * width)), width)
        content_image = cv2.resize(content_image, new_shape)

    (h, w) = content_image.shape[:2]
    print('height:', h)
    print('widht : ', w)

    style_scales = STYLE_SCALE
    if style_scales is not None:
        style_scale = 1.0

        style_images = cv2.resize(style_image, (h, w))
        print('style_images shape : ', style_images.shape)

    style_blend_weights = None
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0/len(style_images) for _ in style_images]
    else:
        #total_blend_weight = sum(style_blend_weights)
        total_blend_weight = style_blend_weights
        style_blend_weights = [weight/total_blend_weight
                               for weight in style_blend_weights]

    initial = None
    initial_noiseblend = None
    if initial is not None:
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])
        # Initial guess is specified, but not noiseblend - no noise should be blended
        if initial_noiseblend is None:
            initial_noiseblend = 0.0
    else:
        # Neither inital, nor noiseblend is provided, falling back to random
        # generated initial guess
        if initial_noiseblend is None:
            initial_noiseblend = 1.0
        if initial_noiseblend < 1.0:
            initial = content_image

    # try saving a dummy image to the output path to make sure that it's writable

    overwrite = True
    if os.path.isfile(output) and not overwrite:
        raise IOError("%s already exists, will not replace it without "
                      "the '--overwrite' flag" % output)
    try:
        imsave(output, np.zeros((500, 500, 3)))
    except:
        raise IOError('%s is not writable or does not have a valid file '
                      'extension for an image file' % output)

    loss_arrs = None
    for iteration, image, loss_vals in stylize(
        network=options.network,
        initial=initial,
        initial_noiseblend=initial_noiseblend,
        content=content_image,
        styles=style_images,
        preserve_colors=False,
        iterations=ITERATIONS,
        content_weight=CONTENT_WEIGHT,
        content_weight_blend=CONTENT_WEIGHT_BLEND,
        style_weight=STYLE_WEIGHT,
        style_layer_weight_exp=STYLE_LAYER_WEIGHT_EXP,
        style_blend_weights=style_blend_weights,
        tv_weight=TV_WEIGHT,
        learning_rate=LEARNING_RATE,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        pooling=POOLING

    ):
        if (image is not None) and (checkpoint_output is not None):
            image = cv2.resize(image, (512, 512))
            imsave(fmt_imsave(checkpoint_output, iteration), image)
        if (loss_vals is not None) \
                and (progress_plot or progress_write):
            if loss_arrs is None:
                itr = []
                loss_arrs = OrderedDict((key, []) for key in loss_vals.keys())
            for key, val in loss_vals.items():
                loss_arrs[key].append(val)
            itr.append(iteration)

    imsave(output, image)

    if progress_write:
        fn = "{}/progress.txt".format(os.path.dirname(output))
        tmp = np.empty((len(itr), len(loss_arrs)+1), dtype=float)
        tmp[:, 0] = np.array(itr)
        for ii, val in enumerate(loss_arrs.values()):
            tmp[:, ii+1] = np.array(val)
        np.savetxt(fn, tmp, header=' '.join(['itr'] + list(loss_arrs.keys())))

    if progress_plot:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        for key, val in loss_arrs.items():
            ax.semilogy(itr, val, label=key)
        ax.legend()
        ax.set_xlabel("iterations")
        ax.set_ylabel("loss")
        fig.savefig("{}/progress.png".format(os.path.dirname(output)))


def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)
    #Image.fromarray(img).save(path, quality=95)


if __name__ == '__main__':
    main()
