# -*- coding: utf-8 -*-
"""Data Processing

"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import datetime, os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, Activation,BatchNormalization,MaxPooling2D,Dropout
from tensorflow.keras.optimizers import Adam

#Load the TensorBoard notebook extension
# %load_ext tensorboard

from google.colab import drive
drive.mount('/content/drive')

#Dataset calling

data_root = Path('/content/drive/MyDrive/BanglaNet')
print('data_root:',data_root)

Dim = 256
batch_size = 10
Num_class = 13

train_steps_per_epoch = 1438 // batch_size
val_steps_per_epoch = 353 // batch_size
test_steps_per_epoch = 154 // batch_size

datagen=ImageDataGenerator(rescale = 1./255, validation_split = 0.2)

train_gen = datagen.flow_from_directory(data_root/'Train',
            target_size = (Dim, Dim),batch_size = 10,subset = 'training')

val_gen = datagen.flow_from_directory(data_root/'Train',
          target_size = (Dim, Dim), batch_size = 10,subset = 'validation')

back2label = {}
for k,v in train_gen.class_indices.items():
    back2label[v] = k
print(back2label)

#Image visulization with labels

x_batch, y_batch = next(train_gen)
for i in range (0, 10):   
  image = x_batch[i]  
  plt.imshow(image)
  print(y_batch[i])
  plt.show()
