#!/usr/bin/env python3
""" Confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix

    Args:
        labels (m, classes): correct labels in one-hot format
        logits (m, classes): predicted labels in one-hot format
    """
    m, classes = labels.shape
    confusion = np.zeros((classes, classes))
    for i in range(m):
        confusion[np.argmax(labels[i]), np.argmax(logits[i])] += 1
    return confusion
