# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:10:37 2022

@author: longc
"""

#%% Import dependencies.
import os
import numpy as np
from matplotlib import pyplot as plt



def train_gan(images, RES_DIR, weights_fiilepath, epochs=1, batch_size=16):
    import time
    iters = epochs * images.shape[0]
    FILE_PATH = '%s/generated_%d.png'
    if not os.path.isdir(RES_DIR):
        os.mkdir(RES_DIR)
    
    CONTROL_SIZE_SQRT = 6
    control_vectors = np.random.normal(size =(CONTROL_SIZE_SQRT**2, LATENT_DIM)) / 2
    
    start = 0
    d_losses = []
    a_losses = []
    images_saved = 0
    for step in range(iters):
      start_time = time.time()    #start timing the execution
    
      # ---------------------
      # Train Discriminator
      # ---------------------
      # Generate a half batch of fake images.
      latent_vectors = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, LATENT_DIM))
      generated = generator.predict(latent_vectors)
    
      # Select a random half batch of real images
      idx = np.random.randint(0, images.shape[0], batch_size)
      real = images[idx]
    
      combined_images = np.concatenate([generated, real])
    
      # fake = 1, real = 0
      labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
      # Label smoothing to discourage the discriminator from being overconfident
      # about its classification.
      labels += 0.05 * np.random.random(labels.shape)
    
      # Train the discriminator network to identify fake and real faces.
      # 'd_loss' is how well the discriminator can tell apart real and fake faces (lower is better)
      d_loss = discriminator.train_on_batch(combined_images, labels)
      d_losses.append(d_loss)
    
      # ------------------
      # Train Generator
      # ------------------
      latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
      # Create misleading labels to try to trick the discriminator to think that the 
      # fakes images are real.
      misleading_targets = np.zeros((batch_size, 1))
    
      # Train the generator network to fool discriminator.
      # 'a_loss' is how well the generator can fool the discriminator (lower value is better)
      a_loss = gan.train_on_batch(latent_vectors, misleading_targets)
      a_losses.append(a_loss)
    
      start += batch_size
      if start > images.shape[0] - batch_size:
        start = 0
      
      if step % 50 == 49:
        gan.save_weights(weights_filepath)
        print('(%d) %d/%d: d_loss: %.4f, a_loss: %.4f (%.1f sec)' % (images_saved, step+1,iters,d_loss,a_loss,time.time()-start_time))
    
        control_image = np.zeros((WIDTH*CONTROL_SIZE_SQRT, HEIGHT*CONTROL_SIZE_SQRT, CHANNELS))
        control_generated = generator.predict(control_vectors)
        for i in range(CONTROL_SIZE_SQRT ** 2):
          x_off = i % CONTROL_SIZE_SQRT
          y_off = i // CONTROL_SIZE_SQRT
          control_image[x_off*WIDTH:(x_off+1)*WIDTH, y_off*HEIGHT:(y_off+1)*HEIGHT, :] = control_generated[i,:,:,:]
        im = Image.fromarray(np.uint8(control_image * 255))
        im.save(FILE_PATH % (RES_DIR, images_saved))
        images_saved += 1
    
      elif step == iters - 1:
        gan.save_weights(weights_filepath)
    
        print('Final iteration reached.\n(%d) %d/%d: d_loss: %.4f, a_loss: %.4f (%.1f sec)' % (images_saved, step+1,iters,d_loss,a_loss,time.time()-start_time))
    
        control_image = np.zeros((WIDTH*CONTROL_SIZE_SQRT, HEIGHT*CONTROL_SIZE_SQRT, CHANNELS))
        control_generated = generator.predict(control_vectors)
        for i in range(CONTROL_SIZE_SQRT ** 2):
          x_off = i % CONTROL_SIZE_SQRT
          y_off = i // CONTROL_SIZE_SQRT
          control_image[x_off*WIDTH:(x_off+1)*WIDTH, y_off*HEIGHT:(y_off+1)*HEIGHT, :] = control_generated[i,:,:,:]
        im = Image.fromarray(np.uint8(control_image * 255))
        im.save(FILE_PATH % (RES_DIR, images_saved))
        images_saved += 1
    
    
    plt.figure(1, figsize=(12,8))
    plt.subplot(121)
    plt.plot(d_losses)
    plt.xlabel('epochs')
    plt.ylabel('discriminant losses')
    plt.subplot(122)
    plt.plot(a_losses)
    plt.xlabel('epochs')
    plt.ylabel('adversary losses')
    plt.show()