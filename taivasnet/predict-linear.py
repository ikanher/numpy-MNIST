#!/usr/bin/env python
"""
Example script of how to do Linear Regression with MSE
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from taivasnet.dataloaders import RegressionDataLoader
from taivasnet.networks import NeuralNet
from taivasnet.models import LinearModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_fname',
                        default='../data/saved_weights_linear.dat',
                        help='Path to the file containing the weights to be used (../data/saved_weights_linear.dat)')

    args = parser.parse_args()
    weights_fname = args.weights_fname

    if not os.path.isfile(weights_fname):
        print("Error: File '{}' does not exist. Did you run `./train.py --save`?".format(weights_fname))
        sys.exit()

    # create the model
    model = LinearModel()
    net = NeuralNet(model=model)
    net.load_weights(weights_fname)

    # we are not training, but predicting
    net.train = False

    # load the data
    ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = RegressionDataLoader().load_data()

    # generate predictions from test set
    predictions = net.forward(x_test)

    print("Test data")
    print("Inputs")
    print(x_test[:10])
    print("Predictions")
    print(predictions[:10])
    print("Actuals")
    print(y_test[:10])

    # generate some random numbers for testing
    randoms = np.random.randint(0, 1000, 10)
    x_rnd = [[x] for x in randoms]
    y_rnd = np.multiply(x_rnd, 2)

    # generate predictions from random numbers
    predictions = net.forward(x_rnd)

    print("Random data")
    print("Inputs")
    print(x_rnd)
    print("Predictions")
    print(predictions)
    print("Actuals")
    print(y_rnd)
