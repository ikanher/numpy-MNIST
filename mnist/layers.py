__author__ = 'Aki Rehn'
__project__ = 'mnist'

import numpy as np

class Linear(object):
    """
    Linear layer with bias
    """

    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        self.weights = self._create_weights(n_input, n_output)
        self.bias = self._create_weights(n_output)

    def _create_weights(self, *dims):
        return np.random.randn(*dims)/dims[0]

    def _create_weights_xavier(self, *dims):
        return np.random.randn(*dims).T / np.sqrt(dims[0] / 2.)

    def forward(self, x):
        """
        Calculates the forward pass, which is just matrix matrix product
        of weights added with bias:  y = xA' + b
        """
        self.x = x
        x = (x @ self.weights) + self.bias
        return x

    def backward(self, grad_output):
        """
        Calculates linear layer gradients
        """
        grad_x = grad_output @ self.weights.T
        grad_weights = (self.x.T @ grad_output)
        grad_bias = np.sum(grad_output, axis=0)

        return grad_x, grad_weights, grad_bias

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
        self.previous = softmax
        return softmax

    def backward(self, y):
        """
        Cross-Entropy Softmax gradient

        https://deepnotes.io/softmax-crossentropy
        """
        grad = self.previous.copy()
        k = grad.shape[0]
        grad[range(k), y] -= 1
        grad = grad/k

        return grad

class Dropout(object):
    """
    Dropout layer

    https://deepnotes.io/dropout
    """

    def __init__(self, p=0.5):
        """
        p is the probability of dropping inputs
        """
        self.p = 1 - p

    def forward(self, x):
        self.mask = np.random.binomial(1, self.p, size=x.shape) / self.p
        return x * self.mask

    def backward(self, grad_output):
        # dropout doesn't need to anything as the dropped weights are
        # dead and don't contribute to the gradient
        return grad_output


class ReLU:
    """
    ReLU (Rectified Linear Unit) layer

    https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """
    def forward(self, x):
        out = x[x > 0]
        self.previous = x
        return out

    def backward(self, grad_output):
        grad = 1.0 * (self.previous > 0) * grad_output
        return grad
