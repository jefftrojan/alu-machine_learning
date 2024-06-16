#!/usr/bin/env python3
""" Sensetivity
"""

import numpy as np


def sensitivity(confusion):
    """ calculates the sensitivity for each class in a confusion matrix

    Args:
        confusion (classes, classes): confusion matrix where row indices
        represent the correct labels and column indices represent the
        predicted labels

    Returns:
        (classes,): sensitivity of each class
    """
    return np.diag(confusion) / np.sum(confusion, axis=1)
