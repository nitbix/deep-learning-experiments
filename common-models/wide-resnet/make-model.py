#!/usr/bin/python
import numpy as np
from keras.layers import Dense, Activation, Flatten, Lambda, Convolution2D, AveragePooling2D, BatchNormalization, Dropout
from keras.engine import merge, Input, Model
from keras.utils import np_utils
import keras.backend as K

nb_classes = 10

img_rows, img_cols = 32, 32
img_channels = 3

blocks_per_group = 4
widening_factor = 10

def residual_block(x, nb_filters=16, subsample_factor=1):
    #make input
    prev_nb_channels = K.int_shape(x)[3]
    input_shape = x.shape
    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shortcut = AveragePooling2D(pool_size=subsample, dim_ordering='tf')(x)
    else:
        subsample = (1, 1)
        # shortcut: identity
        shortcut = x
        
    y = BatchNormalization(axis=3)(x)
    y = Activation('relu')(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=subsample,
                      init='he_normal', border_mode='same', dim_ordering='tf')(y)
    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(y)

    # 1 X 1 conv if shape is different. Else identity.
    if (nb_filters > prev_nb_channels):
        shortcut = Convolution2D(nb_filter=nb_filters, nb_row=1, nb_col=1,
                                 subsample=(1,1),
                                 init="he_normal",
                                 border_mode="same",
                                 dim_ordering='tf'
                                 )(shortcut)

    #make merge
    return merge([y,shortcut], mode="sum")

inputs = Input(shape=(img_rows, img_cols, img_channels))

x = Convolution2D(16, 3, 3, 
                  init='he_normal', border_mode='same', dim_ordering='tf')(inputs)

for i in range(0, blocks_per_group):
    nb_filters = 16 * widening_factor
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=1)

for i in range(0, blocks_per_group):
    nb_filters = 32 * widening_factor
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

for i in range(0, blocks_per_group):
    nb_filters = 64 * widening_factor
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid', dim_ordering='tf')(x)
x = Flatten()(x)

predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(input=inputs, output=predictions)

from pprint import pprint
pprint(model.get_config())

