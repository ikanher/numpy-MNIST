#!/usr/bin/env python

from mnist import dataloader
from mnist.networks import NeuralNet
from mnist.optimizers import SGD
from mnist.losses import CrossEntropy
from mnist.layers import Softmax
from mnist.models import TwoLayerModel

weights_fname = '../data/saved_weights.dat'

# create the model
model = TwoLayerModel(n_input=28*28, n_hidden1=256, n_hidden2=256, n_output=10)
net = NeuralNet(model=model)
net.load_weights(weights_fname)

# create the optimizer
#optimizer = SGD(net=net, dataloader=dataloader.DataLoader(mnist_path='data/'), batch_size=256)
optimizer = SGD(net=net, dataloader=dataloader.DataLoader(), batch_size=512)

# fit the model
optimizer.fit(n_epochs=10, learning_rate=1e-1)
#net.save_weights(weights_fname)

