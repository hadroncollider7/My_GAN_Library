# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:21:23 2022

@author: longc

The purpose of this module is to load and test a generator.
"""
import os 
from define_networks import create_gan 
import tensorflow as tf


def replace_backslash(file_path):
    file_path = file_path.replace(os.sep, '/')


file_path =  "G:\My Drive\Projects\GAN\gan_models\DCGAN\GAN_v3_2_gray_trainSize6878_batchSize16_40epochs.h5"
replace_backslash(file_path)
print(file_path)

#%% Build GAN network.
LATENT_DIM = 32
CHANNELS = 1 
HEIGHT = 128
WIDTH = 128
generator, discriminator, gan = create_gan(CHANNELS=1)

#%% Load weights.
gan.load_weights(file_path)
