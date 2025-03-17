import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import glob
import numpy as np
import tensorflow as tf
import seaborn
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib as plt
import random

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.img import ImageDataGenerator

from sklearn.model_selection import train_test_split

dataset = 'dataset/'
list_label = []

train_folder = 'dataset/train/'
list_label = [folder for folder in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, folder))]

list_folder = ['train', 'test', 'val']

for folder in list_folder:
    for label in list_label:
        path = os.path.join(dataset, folder, label)

image_path = []
labels = []

image_df = pd.DataFrame({'image': image_path, 'labels': labels})

if not image_df.empty:
    train_df, test_df = train_test_split(image_df, test_size=0.2, stratify=image_df['labels'], random_state=42)
else :
    print('data kosong')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./225,
    rotation_range = 10,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./225)

train_generator = train_datagen.flow_from_dataFrame(
    train_df,
    x_col = 'images',
    y_col = 'labels',
    target_size = (150,150),
    batch_size = 32,
    class_mode = 'binary',
    shuffle = False
)

test_generator = test_datagen.flow_from_dataFrame(
    test_df,
    x_col = 'images',
    y_col = 'labels',
    target_size = (150,150)
    batch_size = 32,
    class_mode = 'binary',
    shuffle = False
)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(224, 224, 3)))

model.add(tf.keras.layers.Conv2D(32, (3, 3), acivation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.MaxPooling2D(2, 2))

model.add(tf.keras.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.MaxPooling2D(2, 2))
model.add(tf.keras.GlobalAveragePooling2D())

model.add(tf.keras.Droupout(0.3))
model.add(tf.keras.Dense(128, activation='relu'))
model.add(tf.keras.Dense(128, activation('sigmoid')))

