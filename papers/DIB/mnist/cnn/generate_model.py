#!/usr/bin/python

import keras

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l2

input_shape = [1,28,28]
output_classes = 10

model = Sequential()

model.add(Convolution2D(64, (5, 5),
    input_shape = input_shape,
    bias_regularizer=l2(0.0001), kernel_regularizer=l2(0.0001),
    padding='valid',
    bias_initializer='glorot_normal',
    kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (5, 5),
    bias_regularizer=l2(0.0001), kernel_regularizer=l2(0.0001),
    padding='valid',
    bias_initializer='glorot_normal',
    kernel_initializer='glorot_normal'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(output_classes))
model.add(BatchNormalization(axis=1))
model.add(Activation('softmax'))

print model.to_yaml()
