#!/usr/bin/env python3
""" Early stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ Determines if gradient descent should stop

    Args:
        cost (float): is the current validation cost for the neural network
        opt_cost (float): the lowest recorded cost of the neural network
        threshold (float): the threshold used for early stopping
        patience (float): the patience coutn used for early stopping
        count (int): count of the how long the threshold has not been met.
    Returns: bool whether the network should be stopped early, followed by
    the updated count
        (bool, float)
    """
    if opt_cost - cost <= threshold:
        # if it decrease with an amount less than the threshold, it means
        # the validation cost is still going up,
        count += 1
    else:
        # otherwise the validation cost is going down so we set counter to the
        # patience 0. We start the counter only when current validation cost
        # decrease by an amount less than the threshold (it is not decreasing
        # fast).
        count = 0
    if count < patience:
        return False, count
    return True, count
