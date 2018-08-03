__author__ = 'Aki Rehn'
__project__ = 'mnist'

import numpy as np

class Linear(object):
    """
    Linear layer with bias
    """

    def __init__(self, n_input, n_output):
        self.weights = self._create_weights(n_input, n_output)
        self.bias = self._create_weights(n_output)

    def _create_weights(self, *dims):
        return (np.random.randn(*dims)/dims[0]).T

    def forward(self, x):
        """
        Calculates the forward pass, which is just matrix matrix product
        of weights added with bias: y = xA' + b
        """
        x = (x @ self.weights.T) + self.bias
        return x

    def backward(self, x):
        """
        Calculates linear weights derivative

        TBD
        """
        return

class Softmax(object):
    """
    Softmax layer
    """

    def forward(self, x):
        """
        Calculates Softmax

        https://en.wikipedia.org/wiki/Softmax_function
        """
        return np.exp(x)/np.sum(np.exp(x))

    def backward(self, x):
        """
        Softmax derivative

        TBD
        """
        return

    def cross_entropy(self, y_pred, y):
        """
        Calculates Cross-Entropy Loss

        https://en.wikipedia.org/wiki/Cross_entropy
        """
        out = np.diag(y_pred[:,y])
        return -np.mean(np.log(out))

