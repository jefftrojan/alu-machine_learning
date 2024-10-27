#!/usr/bin/env python3

"""This module creates a sparse autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """This function creates a sparse autoencoder.

    input_dims (int): The dimension of the model input.
    hidden_layers (list): List containing the number of nodes for each
        hidden layer in the encoder. reversed for the decoder.
    latent_dims (int): The dimension of the latent space representation.
    lambtha (float): regularization parameter used for L1 regularization

    Returns:
        tuple: encoder, decoder, auto.
            encoder is the encoder model.
            decoder is the decoder model.
            auto is the full autoencoder model.
    all layers use relu activation except for the decoder output which uses
    sigmoid.
    """
    # encoder
    encoder_input = keras.Input(shape=(input_dims,))
    encoded_output = encoder_input
    for layer in hidden_layers:
        encoded_output = keras.layers.Dense(layer,
                                            activation='relu')(encoded_output)
    l1 = keras.regularizers.l1(lambtha)
    encoded_output = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=l1
    )(encoded_output)
    encoder = keras.models.Model(encoder_input, encoded_output)

    # decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    decoded_output = decoder_input
    for layer in reversed(hidden_layers):
        decoded_output = keras.layers.Dense(layer,
                                            activation='relu')(decoded_output)
    decoded_output = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(decoded_output)
    decoder = keras.models.Model(decoder_input, decoded_output)

    # autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoder_output = encoder(auto_input)
    decoder_output = decoder(encoder_output)
    auto = keras.models.Model(auto_input, decoder_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
