# My_GAN_Library
## Purpose
This is a library to support the GAN model used to generate synthetic dataset of facial emotions. The GAN model uses the Affectnet dataset to train. There are 8 classes of emotions to generate the synthetic dataset on: {0:neutral, 1:happiness, 2:sadness, 3:surprise, 4:fear, 5:disgust, 6:anger, 7:contempt}. 

Currently, I have 8 GAN models, each one trained on one of the classes. Thus, there is a model that can generate faces expressing neutral, another model that generates faces expressing the happy emotions, and so forth. Consequently, there needs to be a way to extract specific classes of images from the Affectnet dataset, and its corresponding "ground truth" labels. This library provides these methods.

## Prerequisites

### Modules
- python = 3.9.12
- tensorflow (with GPU)= 2.9.1
- keras = 2.9.0
- numpy = 1.23.1
- pillow = 9.2.0
- matplotlib = 3.5.1
- seaborn
- pandas