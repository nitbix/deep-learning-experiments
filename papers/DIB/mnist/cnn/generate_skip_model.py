#!/usr/bin/python

import keras

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D, Input
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l2
from keras.models import Model

input_shape = [1,28,28]
output_classes = 10

model = Sequential()

model.add(ZeroPadding2D((1,1), input_shape=input_shape))
model.add(Convolution2D(64, 5, 5, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

input_layer = Input(shape=model.output_shape[1:])
conv = Convolution2D(64,1,1,border_mode='valid')(input_layer)
conv = Convolution2D(64,1,1,border_mode='valid')(conv)
model.add(Model(input=input_layer, output=merge([conv,input_layer], mode="sum")))
model.add(Convolution2D(128, 5, 5, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(output_classes))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('softmax'))

from pprint import pprint
pprint(model.get_config())
