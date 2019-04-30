import sys
import os
import csv
import multiprocessing

import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard


K.clear_session()

tensorboard = TensorBoard(
    log_dir='logs/',
    histogram_freq=0,
    write_graph=True,
    write_images=True,
)

training_data = 'data/train/'
test_data = 'data/test/'

epochs = 2
width, height = 60, 60
batch_size = 20
steps = 500
validation_steps = 200
conv1_filters = 60
conv2_filters = 40
filter1_size = (3, 3)
filter2_size = (2, 2)
pool_size = (2, 2)
classes = 5
learning_rate = 0.001
workers = multiprocessing.cpu_count()
shuffle = True

augmented_data = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    zoom_range=0.1,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = augmented_data.flow_from_directory(
    training_data,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical',
)

test_generator = test_datagen.flow_from_directory(
    test_data,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical',
)


cnn = Sequential()

cnn.add(Convolution2D(conv1_filters, filter1_size, padding="same",
                      input_shape=(width, height, 3), activation='relu'))
cnn.add(Convolution2D(conv1_filters, filter2_size,
                      padding="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Convolution2D(conv2_filters, filter2_size,
                      padding="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dense(256, activation='relu'))
cnn.add(Dense(classes, activation='softmax'))

cnn.summary()

cnn.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(lr=learning_rate),
    metrics=['accuracy'],
)

snn = cnn.fit_generator(
    train_generator,
    steps_per_epoch=steps,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=validation_steps,
    verbose=2,
    shuffle=shuffle,
    workers=workers,
    callbacks=[tensorboard],
)

print(train_generator.class_indices)

target_dir = 'model/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save(os.path.join(target_dir, 'model.h5'))
cnn.save_weights(os.path.join(target_dir, 'weights.h5'))
