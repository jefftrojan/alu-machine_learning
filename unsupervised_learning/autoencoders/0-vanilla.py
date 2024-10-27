#!/usr/bin/env python3

"""This module creates a vanilla autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """This function creates a vanilla autoencoder.

    input_dims (int): The dimension of the model input.
    hidden_layers (list): List containing the number of nodes for each
        hidden layer in the encoder. reversed for the decoder.
    latent_dims (int): The dimension of the latent space representation.

    Returns:
        tuple: encoder, decoder, auto.
            encoder is the encoder model.
            decoder is the decoder model.
            auto is the full autoencoder model.
    all layers use relu activation except for the decoder output which uses
    sigmoid.
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    encoder_output = encoder_input
    for layer in hidden_layers:
        encoder_output = keras.layers.Dense(layer,
                                            activation='relu')(encoder_output)
    encoder_output = keras.layers.Dense(latent_dims,
                                        activation='relu')(encoder_output)
    encoder = keras.models.Model(encoder_input, encoder_output)

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    decoder_output = decoder_input
    for layer in reversed(hidden_layers):
        decoder_output = keras.layers.Dense(layer,
                                            activation='relu')(decoder_output)
    decoder_output = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(decoder_output)
    decoder = keras.models.Model(decoder_input, decoder_output)

    # Autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoder_output = encoder(auto_input)
    decoder_output = decoder(encoder_output)
    auto = keras.models.Model(auto_input, decoder_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
