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
        of weights added with bias:  y = xA' + b
        """
        x = (x @ self.weights.T) + self.bias
        return x

    def backward(self, x, grad_output):
        """
        Calculates linear layer derivative

        """
        grad_weights = (x.T @ grad_output) / x.shape[0]
        grad_bias = np.sum(grad_output, axis=0)

        return grad_weights, grad_bias

class Softmax(object):
    """
    Softmax layer
    """
    def forward(self, x):
        """
        Calculates Softmax

        https://en.wikipedia.org/wiki/Softmax_function
        """
        softmax = np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
        return softmax

    def forward_stable(self, x):
        """
        Calculates Softmax

        https://en.wikipedia.org/wiki/Softmax_function
        """
        z = x - np.max(x, axis=1, keepdims=True)
        exp_z = np.exp(z)
        softmax = exp_z/np.sum(exp_z, axis=1, keepdims=True)
        return softmax

    def backward(self, predictions, y):
        """
        Cross-Entropy Softmax gradient

        https://deepnotes.io/softmax-crossentropy
        """
        k = predictions.shape[0]
        grad = predictions
        grad[range(k), y] -= 1
        grad = grad/k

        return grad

    def cross_entropy(self, y_pred, y):
        """
        Calculates Cross-Entropy Loss

        https://en.wikipedia.org/wiki/Cross_entropy
        """
        predictions = np.diag(y_pred[:,y])
        return -np.mean(np.log(predictions))
