# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:00:53 2022

@author: longc
"""

from keras import Input
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop


def create_generator(LATENT_DIM, CHANNELS, HEIGHT, WIDTH):
  gen_input = Input(shape=(LATENT_DIM,))

  x = Dense(128 * 16 * 16)(gen_input)
  x = LeakyReLU()(x)
  x = Reshape((16, 16, 128))(x)

  x = Conv2D(256, 5, padding='same')(x)
  x = LeakyReLU()(x)

  x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
  x = LeakyReLU()(x)

  x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
  x = LeakyReLU()(x)

  x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
  x = LeakyReLU()(x)

  x = Conv2D(512, 5, padding='same')(x)
  x = LeakyReLU()(x)
  x = Conv2D(512, 5, padding='same')(x)
  x = LeakyReLU()(x)
  x = Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)

  generator = Model(gen_input, x)
  return generator



def create_discriminator(LATENT_DIM, CHANNELS, HEIGHT, WIDTH):
  disc_input = Input(shape=(HEIGHT, WIDTH, CHANNELS))

  x = Conv2D(256, 3)(disc_input)
  x = LeakyReLU()(x)

  x = Conv2D(256, 4, strides=2)(x)
  x = LeakyReLU()(x)

  x = Conv2D(256, 4, strides=2)(x)
  x = LeakyReLU()(x)

  x = Conv2D(256, 4, strides=2)(x)
  x = LeakyReLU()(x)

  x = Conv2D(256, 4, strides=2)(x)
  x = LeakyReLU()(x)

  x = Flatten()(x)
  x = Dropout(0.4)(x)

  x = Dense(1, activation='sigmoid')(x)
  discriminator = Model(disc_input, x)

  optimizer = RMSprop(
    learning_rate = 0.0001,
    clipvalue = 1.0,
    decay = 1e-8
  )

  discriminator.compile(
      optimizer=optimizer,
      loss='binary_crossentropy'
  )
  return discriminator


def compile_gan(gan):
    optimizer = RMSprop(
                    learning_rate=0.0001,
                    clipvalue=1.0,
                    decay=1e-8)
    gan.compile(optimizer=optimizer, loss='binary_crossentropy')
    print(gan.summary())



def create_gan(LATENT_DIM=32, CHANNELS=3, HEIGHT=128, WIDTH=128):
    generator = create_generator(LATENT_DIM, CHANNELS, HEIGHT, WIDTH)
    discriminator = create_discriminator(LATENT_DIM, CHANNELS, HEIGHT, WIDTH)
    discriminator.trainable = False
    
    gan_input = Input(shape=(LATENT_DIM,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    
    print(gan.summary())
    return generator, discriminator, gan
