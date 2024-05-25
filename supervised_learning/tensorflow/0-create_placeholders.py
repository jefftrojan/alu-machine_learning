#!/usr/bin/env python3
"""Class Neuron that defines a single neuron performing binary classification
"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """Function that returns two placeholders, x and y, for the neural network

    Args:
        nx (_type_): _description_
        classes (_type_): _description_
    """
    x = tf.placeholder("float", shape=[None, nx], name="x")
    y = tf.placeholder("float", shape=[None, classes], name="y")
    return x, y
