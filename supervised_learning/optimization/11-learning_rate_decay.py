#!/usr/bin/env python3
""" Learing rate decay
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ updates the learning rate using inverse time decay in numpy

    Args:
        alpha (float): original learning rate
        decay_rate (float): weight used to determine the rate at
        which alpha will decay
        global_step (int): number of passes of gradient descent
        that have elapsed
        decay_step (int): number of passes of gradient descent that
        should occur before alpha is decayed further
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
