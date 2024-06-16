#!/usr/bin/env python3
""" F1 score"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ calculates the F1 score of each class in a confusion matrix

    Args:
        confusion (classes, classes): confusion matrix where row indices
        represent the correct labels and column indices represent the
    Returns:
        (classes,): F1 score of each class
    """
    prec = precision(confusion)
    sens = sensitivity(confusion)
    return 2 * (prec * sens) / (prec + sens)
