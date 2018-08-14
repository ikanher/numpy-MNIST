#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from mnist import dataloader
from mnist.networks import NeuralNet
from mnist.layers import Softmax
from mnist.losses import CrossEntropy
from mnist.models import TwoLayerModel

weights_fname = 'data/saved_weights.dat'

# create the model
model = TwoLayerModel()
net = NeuralNet(model=model)
net.load_weights(weights_fname)
net.train = False

(_, _, (x_test, y_test)) = dataloader.DataLoader('data/').load_data()
test_images = np.reshape(x_test, (-1, 28, 28))

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
