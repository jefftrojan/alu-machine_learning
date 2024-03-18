#!/usr/bin/env python3
"""Function that calculates the likelihood of
obtaining this data given various hypothetical
probabilities of developing severe side effects"""

import numpy as np


def likelihood(x, n, P):
    """Function that calculates the likelihood of
    obtaining this data given various hypothetical
    probabilities of developing severe side effects"""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        text = "x must be an integer that is greater than or equal to 0"
        raise ValueError(text)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if (not isinstance(P, np.ndarray)) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    num = np.math.factorial(n)
    den = np.math.factorial(x) * np.math.factorial(n - x)
    coeficient = num / den
    return coeficient * (P ** x) * ((1 - P) ** (n - x))
