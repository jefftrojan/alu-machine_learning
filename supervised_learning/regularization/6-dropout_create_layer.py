#!/usr/bin/env python3
""" Create a Layer with Dropout"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ Create a Layer with Dropout

    Args:
        prev (_type_): _description_
        n (_type_): _description_
        activation (_type_): _description_
        keep_prob (_type_): _description_
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init)
    dropout = tf.layers.Dropout(rate=keep_prob)
    return dropout(layer(prev))
