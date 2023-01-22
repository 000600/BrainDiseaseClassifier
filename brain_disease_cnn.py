# Imports
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randint

# Define paths
train_path = ' < TRAIN PATH > '
test_path = ' < TEST PATH > '

# Define path specification list
path_list = ['AD', 'CONTROL', 'PD']

# Define classes
classes = ['Alzheimers', 'Normal', 'Parkinsons']

# Initialize lists
x_train = []
y_train = []

x_test = []
y_test = []

def create_dataset(path_stem, path_list):
  x = []
  y = []
  for i in range(len(path_list)):
    paths = []
    string = path_stem + path_list[i]
    for r, d, f in os.walk(fr'{string}'):
        for fi in f:
            if '.jpg' in fi or '.png' in fi or '.jpeg' in fi:
                paths.append(os.path.join(r, fi)) # Add tumor images to the paths list

    # Add images to dataset
    for path in paths:
      img = Image.open(path)
      img = img.resize((128, 128)) # Resize images so that they are easy for the model to understand
      img = np.array(img)
      if (img.shape == (128, 128, 3)):
        x.append(np.array(img))
        y.append(i) # Append corresponding label to y_train

  return x, y

x_train, y_train = create_dataset(train_path, path_list)
x_test, y_test = create_dataset(test_path, path_list)

# Convert dataset into an array
x_train = np.array(x_train)
x_test = np.array(x_test)

# Convert labels into an array
y_train = np.array(y_train)
y_train = y_train.reshape(x_train.shape[0], 1)
y_train = to_categorical(y_train)

y_test = np.array(y_test)
y_test = y_test.reshape(x_test.shape[0], 1)
y_test = to_categorical(y_test)

# View shapes
print('Train Data Shape:', x_train.shape)
print('Train Labels Shape:', y_train.shape)

print('Test Data Shape:', x_test.shape)
print('Test Labels Shape:', y_test.shape)
