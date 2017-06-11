#!/usr/bin/python

import sys
import argparse
import numpy as np
import keras
import theano
from keras import backend as K
from keras.models import load_model
from toupee import data, config

floatX = theano.config.floatX
_TEST_PHASE = np.uint8(0)

def fgm(model, eps=0.3):
    """
    Fast Gradient Sign method.
    :param x: the input placeholder
    :param y: the output placeholder. Use None (the default) to avoid the
            label leaking effect.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf (other norms not implemented yet).
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    """
    #fn = model.model._make_update_inputs_function(eps)

    m = model.model
    inputs = m._feed_inputs + m._feed_targets + m._feed_sample_weights
    if m.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]
    grads = K.gradients(m.total_loss, inputs)
    adv = []
    for p, g, in zip(inputs, grads):
        a = eps * K.sign(g)
        adv.append(a)
    return K.function(inputs = inputs, outputs = adv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate adversarial images and save the numpy arrays')
    parser.add_argument('params_file', help='the parameters file')
    parser.add_argument('model', help='the serialised source model')
    parser.add_argument('--epsilon', type=str, default='0.001', help='epsilon for FastGradientSign method')
    args = parser.parse_args()
    
    epsilon = float(args.epsilon)
    model = load_model(args.model)
    params = config.load_parameters(args.params_file)

    dataset = data.load_data(params.dataset,
                             pickled = params.pickled,
                             one_hot_y = params.one_hot,
                             join_train_and_valid = params.join_train_and_valid,
                             zca_whitening = params.zca_whitening)

    test_set = dataset[2]
    def make_ins(start, end):
        x, y, sample_weights = model.model._standardize_user_data(
                test_set[0][start:end],
                test_set[1][start:end])

        if model.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            ins = x + y + sample_weights + [1.]
        else:
            ins = x + y + sample_weights
        return ins

    adv_f = fgm(model, eps=epsilon)
    
    batch_size = params.batch_size
    n_batches = 1 + len(test_set[1]) / batch_size
    adv = []
    for i in range(0, n_batches):
        start = i * batch_size
        end = min(len(test_set[1]), (i + 1) * batch_size)
        adv_i = adv_f(make_ins(start, end))
        adv.append(adv_i[0])

    adv = np.concatenate(adv).reshape(test_set[0].shape)
 
    if not params.pickled:
        np.savez_compressed(params.dataset + '/adversarial_deltas',
                x=adv,
                y=test_set[1])
        np.savez_compressed(params.dataset + '/adversarial',
                x=test_set[0] + adv,
                y=test_set[1])
    else:
        print "!!! currently cannot save to pickled, saving in local dir"
        np.savez_compressed(model.__name__ + 'adversarial_deltas',
                x=adv,
                y=test_set[1])
        np.savez_compressed('adversarial',
                x=test_set[0] + adv,
                y=test_set[1])
