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

# Set up epochs and batch size
epochs = 30
batch_size = 32

# Initialize SGD Optimizer
opt = SGD(learning_rate = 0.001)

# Create model
model = Sequential()

# Image processing layers
model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
model.add(Conv2D(filters = 30, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = 'valid'))
model.add(Conv2D(filters = 30, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = 'valid'))
model.add(Conv2D(filters = 30, kernel_size = (3, 3), activation = 'relu'))

# Flatten layer
model.add(Flatten())

# Hidden layers
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))

# Output layer
model.add(Dense(3, activation = 'softmax')) # Softmax activation function because the model is a multiclass classifier

# Configure early stopping
early_stopping = EarlyStopping(min_delta = 0.0001, patience = 10, restore_best_weights = True)

# Compile and train model
model.compile(optimizer = opt, loss = CategoricalCrossentropy(), metrics = [CategoricalAccuracy(), AUC()])
history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_test, y_test))


# Visualize  loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plt.plot(loss, label = 'Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize accuracy and validation accuracy
accuracy = history_dict['categorical_accuracy']
val_accuracy = history_dict['val_categorical_accuracy']

plt.plot(accuracy, label = 'Training Accuracy')
plt.plot(val_accuracy, label =' Validation Accuracy')
plt.title('Validation and Training Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualize AUC and validation AUC
auc = history_dict['auc']
val_auc = history_dict['val_auc']

plt.plot(auc, label = 'Training AUC')
plt.plot(val_auc, label = 'Validation AUC')
plt.title('Validation and Training AUC Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.show()

# View test accuracy
test_loss, test_acc, test_auc = model.evaluate(x_test, y_test, verbose = 0) # Change verbose to 1 or 2 for more information
print(f'\nTest accuracy: {test_acc * 100}%')


# View model's predictions compared to actual labels
num_viewed_inputs = 10 # Change this number to view more inputs and corresponding labels and predictions

# Get predictions
predictions = model.predict(x_test)

# Loop through x_test to display the image, the model's prediction on that image, and the actual label of that image
for index in range(num_viewed_inputs):
  # Get random index
  i = randint(0, len(x_test))
  
  # Get image
  image = x_test[i]

  # Model's prediction on sample photo
  predicted_class = np.argmax(predictions[i]) # Get the index with the highest probability assigned to it by the model
  certainty = predictions[i][predicted_class] * 100 # Get assigned probability

  # Actual values
  actual_class = np.argmax(y_test[i])

  # View results
  print(f"\nModel's Prediction ({certainty}% certainty): {predicted_class} ({classes[predicted_class]}) | Actual Class: {actual_class} ({classes[actual_class]})")

  # View input image
  fig = plt.figure(figsize = (3, 3))
  plt.axis('off')
  image_display = plt.imshow(image)
  plt.show(image_display)

# Function for viewing image inputs and the model's predictions based on those image inputs
def display_img(index, predictions_array, true_label, img):
  true_label, img = np.argmax(true_label[index]), img[index]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap = plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'green' # Green text if prediction is correct
  else:
    color = 'red' # Red text if prediction is incorrect

  plt.xlabel(f"{classes[predicted_label]} ({int(100 * np.max(predictions_array))}%) | {classes[true_label]}",color = color) # Return the prediction value and its probability along with the certainty of the prediction in parenthese and the actual value

# Function for displaying multiple predictions and images
def display_array(index, predictions_array, true_label):
  true_label = np.argmax(true_label[index])
  plt.grid(False)
  plt.xticks(range(len(classes)))
  plt.yticks([])
  plot = plt.bar(range(len(classes)), predictions_array, color = "#36454f") # Other prediction probabilites shown in dark gray
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  plot[predicted_label].set_color('red') # Incorrect predictions shown in red
  plot[true_label].set_color('green')# Correct predictions shown in green

# Display incorrect predictions in red, correct predictions in green, and other prediction probabilities in gray
num_rows = 10 # Change this number to view more images and predictions
num_images = num_rows ** 2
plt.figure(figsize = (2 * 2 * num_rows, 2 * num_rows)) # Scale plot to fit all images

# Make a grid with predictions and input images
for ind in range(num_images):
  i = randint(0, len(x_test) - 1)
  plt.subplot(num_rows, 2 * num_rows, 2 * ind + 1)
  display_img(i, predictions[i], y_test, x_test)
  plt.subplot(num_rows, 2 * num_rows, 2 * ind + 2)
  display_array(i, predictions[i], y_test)
  filler = plt.xticks(range(len(classes)), classes, rotation = 90)

# Display plot (the model's prediction probability distribution based on an image is plotted to the right of that image on the plot)
plt.tight_layout()
plt.show()
