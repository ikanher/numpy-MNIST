#!/usr/bin/env python
"""
Makes predictions and displays random images from MNIST test set
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from mnist.dataloaders import MNISTDataLoader
from mnist.networks import NeuralNet
from mnist.models import TwoLayerModel

def print_set_accuracy(net, x, y, msg):
    preds = net.forward(x)
    accuracy = np.mean(preds.argmax(axis=1) == y)
    print(msg, accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_fname',
                        default='../data/saved_weights.dat',
                        help='Path to the file containing the weights to be used (../data/saved_weights.dat)')

    args = parser.parse_args()
    weights_fname = args.weights_fname

    if not os.path.isfile(weights_fname):
        print("Error: File '{}' does not exist. Did you run `./train.py --save`?".format(weights_fname))
        sys.exit()

    # create the model
    model = TwoLayerModel()
    net = NeuralNet(model=model)
    net.load_weights(weights_fname)

    # we are not training, but predicting
    net.train = False

    # load the data
    ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = MNISTDataLoader().load_data()
    test_images = np.reshape(x_test, (-1, 28, 28))

    # print accuracies
    print_set_accuracy(net, x_train, y_train, "Training set accuracy:")
    print_set_accuracy(net, x_valid, y_valid, "Validation set accuracy:")
    print_set_accuracy(net, x_test, y_test, "Test set accuracy:")

    # loop forever to display random images from the set
    while True:
        predictions = net.forward(x_test)
        actuals = y_test

        i = np.random.randint(0, len(x_test))

        fmt = 'Actual: {} - Prediction: {}'
        prediction = np.argmax(predictions[i])
        plt.title(fmt.format(actuals[i], prediction))
        plt.gray()
        plt.imshow(test_images[i])
        plt.show()
