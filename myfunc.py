# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:21:00 2022

@author: longc
"""
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns

from tqdm import tqdm
from PIL import Image
import pandas as pd
import os


def generate_faces(generator, scale=2, LATENT_DIM=32, n=3, cols=10, scatterplot=False):
  """Generate and plot faces."""
  for i in range(3):
    if i < 1:
      v1 = tf.random.normal((30,LATENT_DIM)) / scale
      faces = generator.predict(v1)
    else:
      v1 = tf.random.normal((30,LATENT_DIM)) / scale
      faces = tf.concat([faces, generator.predict(v1)], 0)
  # Fix clipping issues.
  faces = np.clip(faces,0,1)

  rows = faces.shape[0] / cols
  plt.figure(1, figsize=(20, faces.shape[0]/5))
  for i in range(faces.shape[0]):
    plt.subplot(rows,cols,i+1)
    plt.imshow(faces[i])
    plt.axis('off')
  plt.tight_layout()

  # Visualize the latent space distribution.
  if scatterplot == True:
    plt.figure(2)
    sns.scatterplot(v1[0], v1[1])
    plt.xlabel('v1[0]')
    plt.ylabel('v1[1]')
    plt.title('latent dimension')
    plt.grid()
    plt.show()
    
def generate_gray_faces(generator, scale=2, LATENT_DIM=32):
  '''Prints generated grayscale faces.'''
  cols = 10
  for i in range(3):
    if i < 1:
      v1 = tf.random.normal((30,LATENT_DIM)) / scale
      faces = generator.predict(v1)
    else:
      v1 = tf.random.normal((30,LATENT_DIM)) / scale
      faces = tf.concat([faces, generator.predict(v1)], 0)
  # Fix clipping issues.
  faces = np.clip(faces,0,1)
  faces = np.squeeze(faces, 3)
  rows = faces.shape[0] / cols
  plt.figure(1, figsize=(20, faces.shape[0]/5))
  for i in range(faces.shape[0]):
    plt.subplot(rows,cols,i+1)
    plt.imshow(faces[i])
    plt.axis('off')
  plt.tight_layout()
    

def load_affectnet_sample(img_class):
    '''Loads the sample AffectNet dataset from the project directory.
    
    Precondition: img_class is list containing the classes of images
    from the AffectNet dataset that is to be loaded.
    
    '''
    PIC_DIR = '/content/data/train_class/class00'

    SIZE = (128,128)
    
    images = []
    skipcount = 0
    skipimages = []
    for i in range(len(img_class)):
      for pic_file in tqdm(os.listdir(PIC_DIR + f'{img_class[i]}/')):
        if ((pic_file=='image0011604.jpg') or (pic_file=='image0004008.jpg') or (pic_file=='image0003448.jpg') or (pic_file=='image0020054.jpg') or (pic_file=='image0016950.jpg') or (pic_file=='image0036309.jpg') or (pic_file=='image0037576.jpg') or (pic_file=='image0032262.jpg')):
          skipcount += 1
          skipimages.append(pic_file)
        else: 
          pic = Image.open(PIC_DIR + f'{img_class[i]}/' + pic_file)
          pic.thumbnail(SIZE, Image.ANTIALIAS)
          if pic.size != SIZE:
            print('\nSize anomaly: {0:s}, size = {1}'.format(pic_file, pic.size))
            print('at image index: ', len(images))
            images.append(np.uint8(pic))
          else: images.append(np.uint8(pic))
          
    images = np.array(images) / 255   #convert to numpy array and normalize
    
    print('\nNo. of images skipped: ', skipcount)
    print('List of images skipped:')
    for i in range(len(skipimages)):
      print('({0:d}) {1:s}'.format(i+1, skipimages[i]))
    print('\nimages: ')
    print(type(images))
    print('Length: ', len(images))
    print('Element type: ', type(images[0]))
    print('Shape of elements: ', images[0].shape)
    return images



def load_celeba_dataset(IMAGES_COUNT):
    '''Loads the CelebA dataset from the Google Drive directory.
    
    Precondition: IMAGES_COUNT is an integer defining the number of images
    to load. 
    '''
    
    PIC_DIR = '/content/data/img_align_celeba/img_align_celeba/'
    ORIG_WIDTH = 178
    ORIG_HEIGHT = 208
    diff = (ORIG_HEIGHT - ORIG_WIDTH) // 2    # '//' is a floor division
    SIZE = (128,128)
    
    crop_rect =(0, diff, ORIG_WIDTH, ORIG_HEIGHT - diff)
    images = []
    for pic_file in tqdm(os.listdir(PIC_DIR)[:IMAGES_COUNT]):
        pic = Image.open(PIC_DIR + pic_file).crop(crop_rect)
        pic.thumbnail(SIZE, Image.ANTIALIAS)
        if pic.size != SIZE:
          print('\nSize anomaly: {0:s}, size = {1}'.format(pic_file, pic.size))
          print('at image index: ', len(images))
          images.append(np.uint8(pic))
        else: images.append(np.uint8(pic))
    
    images = np.array(images) / 255
    
    print('\nimages: ')
    print(type(images))
    print('Length: ', len(images))
    print('Element type: ', type(images[0]))
    print('Shape of elements: ', images[0].shape)
    return images


def get_image_info(images):
    '''Prints image class, shape, datatype of elements, and minmax values'''
    print(type(images))
    print('images shape: ', np.asarray(images).shape)
    print('images element datatype: ', images[0].dtype)
    print('images as array datatype: ', np.asarray(images).dtype)
    print('minmax: ', np.amin(images),np.amax(images))

















