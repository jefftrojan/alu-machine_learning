#!/usr/bin/env python3
""" Precision
"""

import numpy as np


def precision(confusion):
    """ calculates the precision for each class in a confusion matrix

    Args:
        confusion (classes, classes): confusion matrix where row indices
        represent the correct labels and column indices represent the
        predicted labels

    Returns:
        (classes,): precision of each class
    """
    return np.diag(confusion) / np.sum(confusion, axis=0)
