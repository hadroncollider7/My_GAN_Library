# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 15:48:20 2022

@author: longc
"""

import numpy as np 
import tarfile
import os
from tqdm import tqdm
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import pickle
import time

# Funtions
def extract_tar(file_path, save_to_path):
    '''Extracts tar file from a filepath.'''
    with tarfile.open(file_path, mode='r') as tar:
        tar.extractall(save_to_path)
        print('Done')
    
# Extract expression labels.
def extract_expression_labels(PATH):
    '''
    Extracts the labels for the facial expression classes.
    
    Returns:
        Dictionary containing the labels for expression.
    '''
        
    expr_labels = {}
    for filename in tqdm(os.listdir(PATH)):
        if filename.find('exp') != -1:
            expr_labels[filename] = np.load(PATH + filename)
    print('Length of expression labels: ', len(expr_labels))
    return expr_labels



def savelist(list_var, file_path):
    '''Saves a list or dictionary.'''
    filehandler = open(file_path, 'wb')
    pickle.dump(list_var, filehandler)
    filehandler.close()
        
def loadlist(file_path):
    '''Loads a list or dictionary.
    
    Returns:
        A list or dictionary.'''
    filehandler = open(file_path, 'rb')
    list_var = pickle.load(filehandler)
    return list_var

    
def extract_images(PIC_DIR, SIZE, grayscale=False):
    images = []
    for pic_file in tqdm(os.listdir(PIC_DIR)):
        for idx in range(len(class_filenames)):    
            if (pic_file[:len(pic_file) - 4]) == class_filenames[idx][:len(class_filenames[idx]) - 8]:
                pic = Image.open(PIC_DIR + pic_file)
                
                if grayscale == False:
                    pic.thumbnail(SIZE, Image.ANTIALIAS)
                    if pic.size != SIZE:
                      print('\nSize anomaly: {0:s}, size = {1}'.format(pic_file, pic.size))
                      print('skipping over: ', pic_file)
                    else: images.append(np.uint8(pic))
                else:
                    pic = ImageOps.grayscale(pic)
                    pic.thumbnail(SIZE, Image.ANTIALIAS)
                    if pic.size != SIZE:
                      print('\nSize anomaly: {0:s}, size = {1}'.format(pic_file, pic.size))
                      print('skipping over: ', pic_file)
                    else: images.append(np.uint8(pic))
    return images



#%% Extract images from tar file.
if __name__ == "__main__":
    file_path = "D:/Projects/Datasets/AffectNet/val_set.tar"
    save_to_path = 'data/'
    extract_tar(file_path, save_to_path)
    
    # Extract expression labels from the annotations.
    expression_labels = extract_expression_labels('data/val_set/annotations/')
    
    #%%
    #**************************************************
    # Extract a class of expression from the dataset:
    # 0: neutral
    # 1: happiness
    # 2: sadness
    # 3: surprise
    # 4: fear
    # 5: disgust
    # 6: anger
    # 7: contempt
    #**************************************************
    # *********** LOAD EXPRESSION LABELS **************
    trainingLabels = False      # SelectS labels of training images or validation images.
    
    if not trainingLabels:
        expression_labels = loadlist('data/saved_objects/expression_labels_validationset')
    else:
        expression_labels = loadlist('data/saved_objects/expression_labels_trainset')
    
    # Separte keys and values.
    key_list = list(expression_labels.keys())     #the filename
    val_list = list(expression_labels.values())   #the corresponding label
    
    
    #%% Define the class of emotion to be extracted from the dataset.
    expr_class = 1
    
    # Extract class indices from emotion labels.
    class_label_idx = []
    for i in range(len(val_list)):
        if int(val_list[i]) == expr_class:
            class_label_idx.append(i)
            
    # Extract fear emotion filenames from the dataset.
    class_filenames = [key_list[i] for i in class_label_idx]
    
    
    #%% Extract images belonging to the desired class of emotions.
    start_time = time.time()
    if not trainingLabels:
        PIC_DIR = 'data/val_set/images/'        #directory to extract the images from
    else:
        PIC_DIR = 'data/train_set/train_set/images/'
    SIZE = (128, 128)
    images = extract_images(PIC_DIR, SIZE, grayscale=True)
    # Normalize images.  
    try:      
        images = np.array(images) / 255
        print('Images normalized.')
    except:
        print('Images were unable to be normalized.')
    
    print('\nimages: ')
    print(type(images))
    print('Length: ', len(images))
    print('Element type: ', type(images[0]))
    print('Shape of elements: ', images[0].shape)
    print('{0:.3f} sec'.format(time.time() - start_time))
    
    
    
    #%% Print a sample of the images.
    rng = np.random.randint(0, images.shape[0], size=(100), dtype=np.uint32)
    plt.figure(1, figsize=(20,20))
    for i in range(100):
      plt.subplot(10, 10, i+1, xlabel=rng[i])
      plt.gray()
      plt.imshow(images[rng[i]])
      plt.xticks([])
      plt.yticks([])
    plt.show
    #%%
    print(type(images))
    print(images.shape)
    print(images.dtype)
    print('minmax: ', np.amin(images), np.amax(images))
    
    #%% Save the images and np arrays.
    if not trainingLabels:
        np.save('data/saved_objects/class00{0:d}_valset_gray.npy'.format(expr_class), images)
    else:
        np.save('data/saved_objects/class00{0:d}_gray.npy'.format(expr_class), images)
    
    
    



















