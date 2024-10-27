#!/usr/bin/env python3

"""This module creates a convolutional autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """This function creates a convolutional autoencoder.

    input_dims (tuple of ints): The dimension of the model input.
    filters (list): List of the number of filters for each convolutional
        layer in the encoder. reversed for the decoder.
    latent_dims (tuple of int): dimension of the latent space representation.
    - convolutions should have (3,3) kernel size with same padding and
    relu activation and upsampling of size (2, 2)
    - 2nd to last convolution should use valid padding
    - last convolution should have the same number of channels as the
    original input with sigmoid activation and no upsampling

    Returns:
        tuple: encoder, decoder, auto.
            encoder is the encoder model.
            decoder is the decoder model.
            auto is the full autoencoder model.
    all layers use relu activation except for the decoder output which uses
    sigmoid.
    """
    # encoder
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3),
                                activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.models.Model(encoder_input, x)

    # decoder
    encoded_shape = keras.backend.int_shape(x)[1:]
    decoder_input = keras.Input(shape=encoded_shape)
    x = decoder_input
    for f in filters[::-1]:
        x = keras.layers.Conv2D(f, (3, 3),
                                activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(input_dims[-1], (3, 3),
                            activation='sigmoid', padding='same')(x)
    # Adjust the final output size
    x = keras.layers.Cropping2D(((2, 2), (2, 2)))(x)
    decoder = keras.models.Model(decoder_input, x)

    # autoencoder
    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.models.Model(auto_input, decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
