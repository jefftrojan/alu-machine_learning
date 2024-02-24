#!/usr/bin/env python3
"""Function that performs a valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images
    Args:
        images: `numpy.ndarray` with shape (m, h, w)
            containing multiple grayscale images
            m: `int`, is the number of images
            h: `int`, is the height in pixels of the images
            w: `int`, is the width in pixels of the images
        kernel: `numpy.ndarray` with shape (kh, kw)
            containing the kernel for the convolution
            kh: `int`, is the height of the kernel
            kw: `int`, is the width of the kernel
    Returns:
        output: `numpy.ndarray` containing the convolved images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    nw = w - kw + 1
    nh = h - kh + 1
    convolved = np.zeros((m, nh, nw))
    for i in range(nh):
        for j in range(nw):
            image = images[:, i:(i + kh), j:(j + kw)]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2))
    return convolved
