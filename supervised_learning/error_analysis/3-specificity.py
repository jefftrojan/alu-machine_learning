#!/usr/bin/env python3
""" Specificity
"""

import numpy as np


def specificity(confusion):
    """ calculates the specificity for each class in a confusion matrix

    Args:
        confusion (classes, classes): confusion matrix where row indices
        represent the correct labels and column indices represent the
        predicted labels

    Returns:
        (classes,): specificity of each class
    """
    true_pos = np.diag(confusion)
    false_neg = np.sum(confusion, axis=1) - true_pos
    false_pos = np.sum(confusion, axis=0) - true_pos
    true_neg = np.sum(confusion) - (true_pos + false_neg + false_pos)
    return true_neg / (true_neg + false_pos)
