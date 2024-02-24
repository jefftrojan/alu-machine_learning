#!/usr/bin/env python3
"""Function that performs a convolution on images using multiple kernels"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels
    Args:
        images: `numpy.ndarray` with shape (m, h, w)
            containing multiple grayscale images
            m: `int`, is the number of images
            h: `int`, is the height in pixels of the images
            w: `int`, is the width in pixels of the images
            c: `int`, is the number of channels in the image
        kernels: `numpy.ndarray` with shape (kh, kw, c, nc)
            containing the kernel for the convolution
            kh: `int`, is the height of the kernel
            kw: `int`, is the width of the kernel
            nc: `int`, is the number of kernels
        padding: `tuple` of (ph, pw), ‘same’, or ‘valid’
            if `tuple`:
                ph: `int` is the padding for the height of the image
                pw: `int` is the padding for the width of the image
            if ‘same’, performs a same convolution
            if ‘valid’, performs a valid convolution
        stride is a tuple of (sh, sw)
            sh: `int`, is the stride for the height of the image
            sw: `int`, is the stride for the width of the image
    Returns:
        output: `numpy.ndarray` containing the convolved images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw, nc = kernels.shape[0], kernels.shape[1], kernels.shape[3]
    sh, sw = stride[0], stride[1]
    if padding == 'same':
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        pw = padding[1]
        ph = padding[0]
    nw = int(((w - kw + (2 * pw)) / sw) + 1)
    nh = int(((h - kh + (2 * ph)) / sh) + 1)
    convolved = np.zeros((m, nh, nw, nc))
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(nh):
        x = i * sh
        for j in range(nw):
            y = j * sw
            for k in range(nc):
                image = imagesp[:, x:x + kh, y:y + kw, :]
                kernel = kernels[:, :, :, k]
                convolved[:, i, j, k] = np.sum(np.multiply(image, kernel),
                                               axis=(1, 2, 3))
    return convolved
