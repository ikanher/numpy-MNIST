"""
Contains different layers to be used in Neural Network
"""

__author__ = 'Aki Rehn'
__project__ = 'taivasnet'

import numpy as np

class Layer():
    """
    Abstract class representing a layer in network
    """

    def __init__(self):
        # should this layer be used in training phase only
        self.train_only = False

        # is this layer learning? ie. should we store the gradients
        self.learning = False

        # this is where the gradients will be stored, if required
        self.grad_w = None
        self.grad_b = None

    def forward(self, x):
        """
        Abstract method for computing the forward pass
        """
        raise NotImplementedError("forward not implemented")

    def backward(self, grad_output):
        """
        Abstract method for computing the backward pass
        """
        raise NotImplementedError("backward not implemented")

class Linear(Layer):
    """
    Linear layer with bias
    """

    def __init__(self, n_input, n_output):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.weights = self._create_weights(n_input, n_output)
        self.bias = self._create_weights(n_output)
        self.learning = True
        self.x = None

    def _create_weights(self, *dims):
        return np.random.randn(*dims)/dims[0]

    def _create_weights_xavier(self, *dims):
        return np.random.randn(*dims) / np.sqrt(dims[0] / 2.)

    def forward(self, x):
        """
        Calculates the forward pass, which is just matrix matrix product
        of weights added with bias: y = xA + b
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

class Softmax(Layer):
    """
    Softmax layer
    """

    def __init__(self):
        super().__init__()
        self.previous = None

    def forward(self, x):
        """
        Calculates Softmax

        Basically return values that sum up to 1 and can be used as
        probabilities.

        https://en.wikipedia.org/wiki/Softmax_function
        """
        softmax = np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
        self.previous = softmax
        return softmax

    def backward(self, y):
        """
        Cross-Entropy Softmax gradient

        Calculates the difference between actual values and predictions.
        This is also the derivative of the Cross Entropy loss function.

        https://deepnotes.io/softmax-crossentropy
        """
        grad = self.previous.copy()
        k = grad.shape[0]
        grad[range(k), y] -= 1
        grad = grad/k

        return grad

class Dropout(Layer):
    """
    Dropout layer

    Simply drops some of the inputs according to a threshold.

    It is used as regularization method.

    https://deepnotes.io/dropout
    """

    def __init__(self, p=0.5):
        """
        p is the probability of dropping inputs
        """
        super().__init__()
        self.train_only = True
        self.p = 1 - p

    def forward(self, x):
        mask = np.random.binomial(1, self.p, size=x.shape) / self.p
        return x * mask

    def backward(self, grad_output):
        # dropout doesn't need to anything as the dropped weights are
        # dead and don't contribute to the gradient
        return grad_output


class ReLU(Layer):
    """
    ReLU (Rectified Linear Unit) layer

    If x is greater than zero returns x, else 0.

    https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """
    def __init__(self):
        super().__init__()
        self.previous = None

    def forward(self, x):
        out = np.maximum(0, x)
        self.previous = x
        return out

    def backward(self, grad_output):
        grad = (self.previous > 0) * grad_output
        return grad
