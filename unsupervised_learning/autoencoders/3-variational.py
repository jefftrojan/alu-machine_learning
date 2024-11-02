#!/usr/bin/env python3
"""
Defines function that creates a variational autoencoder
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    """
    if type(input_dims) is not int:
        raise TypeError(
            "input_dims must be an int containing dimensions of model input")
    if type(hidden_layers) is not list:
        raise TypeError("hidden_layers must be a list of ints \
        representing number of nodes for each layer")
    for nodes in hidden_layers:
        if type(nodes) is not int:
            raise TypeError("hidden_layers must be a list of ints \
            representing number of nodes for each layer")
    if type(latent_dims) is not int:
        raise TypeError("latent_dims must be an int containing dimensions of \
        latent space representation")

    # encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    encoder_value = encoder_inputs
    for i in range(len(hidden_layers)):
        encoder_layer = keras.layers.Conv2D(hidden_layers[i],
                                            activation='relu',
                                            kernel_size=(3, 3),
                                            padding='same')
        encoder_value = encoder_layer(encoder_value)
        encoder_batch_norm = keras.layers.BatchNormalization()
        encoder_value = encoder_batch_norm(encoder_value)
    encoder_flatten = keras.layers.Flatten()
    encoder_value = encoder_flatten(encoder_value)
    encoder_dense = keras.layers.Dense(activation='relu')
    encoder_value = encoder_dense(encoder_value)
    encoder_batch_norm = keras.layers.BatchNormalization()
    encoder_value = encoder_batch_norm(encoder_value)

    encoder_output_layer = keras.layers.Dense(units=latent_dims,
                                              activation='relu')
    encoder_outputs = encoder_output_layer(encoder_value)
    encoder = keras.Model(inputs=encoder_inputs, outputs=encoder_outputs)

    # decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    decoder_value = decoder_inputs
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoder_layer = keras.layers.Dense(units=hidden_layers[i],
                                           activation='relu')
        decoder_value = decoder_layer(decoder_value)
    decoder_output_layer = keras.layers.Dense(units=input_dims,
                                              activation='sigmoid')
    decoder_outputs = decoder_output_layer(decoder_value)
    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # autoencoder
    inputs = encoder_inputs
    auto = keras.Model(inputs=inputs, outputs=decoder(encoder(inputs)))
    auto.compile(optimizer='adam',
                 loss='binary_crossentropy')

    return encoder, decoder, auto
