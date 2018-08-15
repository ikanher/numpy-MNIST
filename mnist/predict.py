#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from mnist import dataloader
from mnist.networks import NeuralNet
from mnist.layers import Softmax
from mnist.losses import CrossEntropy
from mnist.models import TwoLayerModel

def print_set_accuracy(net, x, y, msg):
    preds = net.forward(x)
    accuracy = np.mean(preds.argmax(axis=1) == y)
    print(msg, accuracy)

weights_fname = '../data/saved_weights.dat'

# create the model
#model = TwoLayerModel()
model = TwoLayerModel(n_input=28*28, n_hidden1=256, n_hidden2=256, n_output=10)
net = NeuralNet(model=model)
net.load_weights(weights_fname)
net.train = False

((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = dataloader.DataLoader().load_data()
test_images = np.reshape(x_test, (-1, 28, 28))

print_set_accuracy(net, x_train, y_train, "Training set accuracy:")
print_set_accuracy(net, x_valid, y_valid, "Validation set accuracy:")
print_set_accuracy(net, x_test, y_test, "Test set accuracy:")

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

