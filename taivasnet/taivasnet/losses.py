"""
Module contains loss functions to be used in a Neural Network
"""

__author__ = 'Aki Rehn'
__project__ = 'taivasnet'

import numpy as np

class CrossEntropy():
    """
    Cross-Entropy Loss

    https://en.wikipedia.org/wiki/Cross_entropy
    """
    def loss(self, y_pred, y):
        """
        Calculates Cross-Entropy Loss

        y_pred - predictions
        y - target values
        """
        predictions = np.diag(y_pred[:, y])
        return -np.mean(np.log(predictions))

    def gradient(self, y_pred, y, inputs):
        """
        Cross-Entropy Softmax gradient

        Calculates the difference between actual values and predictions.
        This is also the derivative of the Cross Entropy loss function.

        https://deepnotes.io/softmax-crossentropy

        y_pred - predictions
        y - target values
        inputs - not used where
        """
        grad = y_pred.copy()
        k = grad.shape[0]
        grad[range(k), y] -= 1
        grad = grad/k
        return grad

class MSE():
    """
    Mean Squared Error

    https://en.wikipedia.org/wiki/Mean_squared_error
    """

    def loss(self, y_pred, y):
        """
        Calculates Mean Squared Error loss

        y_pred - predictions
        y - target values
        """
        return np.sum((y_pred - y)**2) / y.size

    def gradient(self, y_pred, y, inputs):
        return 2*(y_pred - y) / y.size
