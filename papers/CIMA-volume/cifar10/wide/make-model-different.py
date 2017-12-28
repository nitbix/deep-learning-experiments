from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import os

import logging
logging.basicConfig(level=logging.DEBUG)

import sys
# Prevent reaching to maximum recursion depth in `theano.tensor.grad`
sys.setrecursionlimit(2 ** 20)

import numpy as np
np.random.seed(2 ** 10)

from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense, Flatten, add
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K

# ================================================
# DATA CONFIGURATION:
logging.debug("Loading data...")

nb_classes = 10
image_size = 32

# ================================================
# NETWORK/TRAINING CONFIGURATION:
logging.debug("Loading network/training configuration...")

depth = 28              # table 5 on page 8 indicates best value (4.17) CIFAR-10
k = 10                  # 'widen_factor'; table 5 on page 8 indicates best value (4.17) CIFAR-10
dropout_probability = 0 # table 6 on page 10 indicates best value (4.17) CIFAR-10

weight_decay = 0.0005   # page 10: "Used in all experiments"

batch_size = 128        # page 8: "Used in all experiments"
# Regarding nb_epochs, lr_schedule and sgd, see bottom page 10:
nb_epochs = 200
lr_schedule = [60, 120, 160] # epoch_step
def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02 # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

# Other config from code; throughtout all layer:
use_bias = False        # following functions 'FCinit(model)' and 'DisableBias(model)' in utils.lua
weight_init="he_normal" # follows the 'MSRinit(model)' function in utils.lua

# Keras specific
if K.image_dim_ordering() == "th":
    logging.debug("image_dim_ordering = 'th'")
    channel_axis = 1
    input_shape = (3, image_size, image_size)
else:
    logging.debug("image_dim_ordering = 'tf'")
    channel_axis = -1
    input_shape = (image_size, image_size, 3)
# ================================================

# ================================================
# OUTPUT CONFIGURATION:
print_model_summary = True
save_model_and_weights = True
save_model_plot = False

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/')
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
# ================================================


# Wide residual network http://arxiv.org/abs/1605.07146
def _wide_basic(n_input_plane, n_output_plane, stride):
    def f(net):
        # format of conv_params:
        #               [ [nb_col="kernel width", nb_row="kernel height",
        #               subsample="(stride_vertical,stride_horizontal)",
        #               border_mode="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [ [3,3,stride,"same"],
                        [3,3,(1,1),"same"] ]
        
        n_bottleneck_plane = n_output_plane

        input_layer = Input(shape=net.layers[-1].output_shape[1:])
        net = input_layer
        # Residual block
        for i, v in enumerate(conv_params):
            if i == 0:
                if n_input_plane != n_output_plane:
                    net = BatchNormalization(axis=channel_axis)(net)
                    net = Activation("relu")(net)
                    convs = net
                else:
                    convs = BatchNormalization(axis=channel_axis)(net)
                    convs = Activation("relu")(convs)
                convs = Convolution2D(n_bottleneck_plane,
                                     kernel_size=(v[0],v[1]),
                                     strides=v[2],
                                     padding=v[3],
                                     kernel_initializer=weight_init,
                                     bias_initializer=weight_init,
                                     kernel_regularizer=l2(weight_decay),
                                     use_bias=use_bias)(convs)
            else:
                convs = BatchNormalization(axis=channel_axis)(convs)
                convs = Activation("relu")(convs)
                if dropout_probability > 0:
                   convs = Dropout(dropout_probability)(convs)
                convs = Convolution2D(n_bottleneck_plane,
                                     kernel_size=(v[0],v[1]),
                                     strides=v[2],
                                     padding=v[3],
                                     kernel_initializer=weight_init,
                                     bias_initializer=weight_init,
                                     kernel_regularizer=l2(weight_decay),
                                     use_bias=use_bias)(convs)

        # Shortcut Conntection: identity function or 1x1 convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut = Convolution2D(n_output_plane,
                                     kernel_size=(1,1),
                                     strides=stride,
                                     padding="same",
                                     kernel_initializer=weight_init,
                                     bias_initializer=weight_init,
                                     kernel_regularizer=l2(weight_decay),
                                     use_bias=use_bias)(net)
        else:
            shortcut = net

        merge_layer = add([convs, shortcut])
        return Model(inputs=[input_layer],outputs=[merge_layer])
    return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, count, stride, model):
        model.add(block(n_input_plane, n_output_plane, stride)(model))
        for i in range(2,int(count+1)):
            model.add(block(n_output_plane, n_output_plane, stride=(1,1))(model))


def create_model():
    logging.debug("Creating model...")
    
    assert((depth - 4) % 6 == 0)
    n = (depth - 4) / 6
    
    model = Sequential()

    n_stages=[16, 16*k, 32*k, 64*k]

    conv1 = Convolution2D(filters=n_stages[0],
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          input_shape=input_shape,
                          padding="same",
                          kernel_initializer=weight_init,
                          bias_initializer=weight_init,
                          kernel_regularizer=l2(weight_decay),
                          use_bias=use_bias) # "One conv at the beginning (spatial size: 32x32)"

    model.add(conv1)
    # Add wide residual blocks
    block_fn = _wide_basic
    _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1,1), model=model)# "Stage 1 (spatial size: 32x32)"
    _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2,2), model=model)# "Stage 2 (spatial size: 16x16)"
    _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2,2), model=model)# "Stage 3 (spatial size: 8x8)"

    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation("relu"))
                                            
    # Classifier block
    pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")
    model.add(pool)
    model.add(Flatten())
    predictions = Dense(units=nb_classes,
                        kernel_initializer=weight_init,
                        bias_initializer=weight_init,
                        use_bias=use_bias,
                        kernel_regularizer=l2(weight_decay), activation="softmax")
    model.add(predictions)
    return model


if __name__ == '__main__':
    model = create_model()
    print(model.to_yaml())
