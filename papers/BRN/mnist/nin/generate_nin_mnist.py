import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l2

input_shape = [3,40,40]
output_classes = 10

model = Sequential()

model.add(ZeroPadding2D((2,2), input_shape=input_shape))
model.add(Convolution2D(192, 5, 5, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Convolution2D(160, 1, 1, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Convolution2D(96, 1, 1, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(192, 5, 5, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Convolution2D(192, 1, 1, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Convolution2D(192, 1, 1, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Convolution2D(192, 5, 5, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Convolution2D(192, 1, 1, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Convolution2D(10, 1, 1, b_regularizer=l2(0.0001), W_regularizer=l2(0.0001), border_mode='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(mode=2,axis=1))

model.add(Flatten())
model.add(Dense(output_classes))
model.add(BatchNormalization(mode=2,axis=1))
model.add(Activation('softmax'))
print model.to_yaml()
