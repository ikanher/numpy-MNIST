"""
Module contains Neural Network implementations
"""

import numpy as np

__author__ = 'Aki Rehn'
__project__ = 'taivasnet'

class NeuralNet():
    """
    Class representing a neural network
    """

    def __init__(self, model):
        """
        Constructor

        model - model to be used
        """
        self.train = True
        self.layers = model.get_layers()
        self.loss_func = model.get_loss_func()

    def forward(self, x):
        """
        Forward propagates through the network
        """
        for layer in self.layers:

            if self.train:
                x = layer.forward(x)
            else:
                if not layer.train_only:
                    x = layer.forward(x)

        return x

    def backward(self, grad_loss):
        """
        Backpropagates through the network

        https://en.wikipedia.org/wiki/Backpropagation
        """

        grad_output = grad_loss

        # iterate through the rest of the layers in reverse order
        for layer in reversed(self.layers):

            if layer.learning:
                grad_output, grad_w, grad_b = layer.backward(grad_output)
                layer.grad_w = grad_w
                layer.grad_b = grad_b
            else:
                grad_output = layer.backward(grad_output)

        return grad_output


    def loss(self, y_pred, y):
        """
        Calculates loss from predictions and targets using loss_func

        y_pred - prediction
        y - target values
        """
        return self.loss_func.loss(y_pred, y)

    def loss_gradient(self, y_pred, y, inputs):
        """
        Calculates the gradient of the loss function

        y_pred - prediction
        y - target values
        inputs - inputs to the network
        """
        return self.loss_func.gradient(y_pred, y, inputs)

    def save_weights(self, filename):
        """
        Saves the model weights into a file
        """
        with open(filename, 'wb') as f:
            for layer in self.layers:
                if not layer.learning:
                    continue

                np.save(f, layer.weights)
                np.save(f, layer.bias)

    def load_weights(self, filename):
        """
        Loads model weights from a file
        """
        with open(filename, 'rb') as f:
            for layer in self.layers:
                if not layer.learning:
                    continue

                w = np.load(f)
                if w.shape != layer.weights.shape:
                    msg = "Cannot load layer weights with different shapes '{}' and '{}'."
                    msg += ' Did you run `train.py --save`?'
                    raise RuntimeError(msg.format(w.shape, layer.weights.shape))

                layer.weights = w

                b = np.load(f)
                if b.shape != layer.bias.shape:
                    msg = "Cannot load bias weights with different shapes '{}' and '{}'"
                    msg += ' Did you run `train.py --save`?'
                    raise RuntimeError(msg.format(b.shape, layer.bias.shape))

                layer.bias = b
