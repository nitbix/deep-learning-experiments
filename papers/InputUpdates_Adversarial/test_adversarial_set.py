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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate adversarial images and save the numpy arrays')
    parser.add_argument('params_file', help='the parameters file')
    parser.add_argument('model', help='the serialised source model')
    parser.add_argument('--epsilons', type=str, default='0.001', help='epsilon list for FastGradientSign method')
    args = parser.parse_args()
    
    params = config.load_parameters(args.params_file)
    epsilons = [float(x) for x in args.epsilons.split(',')]

    model = load_model(args.model)
    t = data.load_single_file(params.dataset + 'test',
                              one_hot_y = params.one_hot)
    d = data.load_single_file(params.dataset + 'adversarial_deltas')
    d = (np.sign(d[0]), d[1])

    for epsilon in epsilons:
        metrics = model.evaluate(t[0] + epsilon * d[0], d[1], batch_size=params.batch_size)
        for i in range(len(metrics)):
            print "Epsilon = {0}:  {1} = {2}".format(epsilon, model.metrics_names[i], metrics[i])
