"""
Module contains models to be used in neural networks
"""

__author__ = 'Aki Rehn'
__project__ = 'taivasnet'

from taivasnet.losses import CrossEntropy, MSE
from taivasnet.layers import Linear, Dropout, ReLU, Softmax

class Model():
    """
    Abstract class representing neural network models
    """

    def get_layers(self):
        raise NotImplementedError("get_layers not implemented")

    def get_loss_func(self):
        raise NotImplementedError("get_loss_func not implemented")

class LinearModel(Model):
    """
    Linear model
    """

    def __init__(self, n_input=1, n_hidden=10, n_output=1):
        """
        Construct a new LinearModel
        """
        # model parameters
        self.n_input = n_input
        self.n_hidden = 10
        self.n_output = n_output

    def get_layers(self):
        """
        Returns a list of layers that make this model
        """
        layers = []
        layers.append(Linear(self.n_input, self.n_hidden))
        layers.append(ReLU())
        layers.append(Linear(self.n_hidden, self.n_output))

        return layers

    def get_loss_func(self):
        """
        Returns the loss function to be used
        """
        return MSE()

class TwoLayerModel(Model):
    """
    Model containing two hidden layers with Softmax output and Cross-Entropy loss
    """

    def __init__(self, n_input=28*28, n_hidden1=256, n_hidden2=64, n_output=10):
        """
        Construct a new TwoLayerModel

        n_inputs - number of inputs
        n_hidden1 - number of nodes in the first hidden layer
        n_hidden2 - number of nodes in the second hidden layer
        n_output - number of outputs
        """
        # model parameters
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output

    def get_layers(self):
        """
        Returns a list of layers that make this model
        """
        layers = []
        layers.append(Dropout(0.01))                            # regularization
        layers.append(Linear(self.n_input, self.n_hidden1))     # input layers
        layers.append(Dropout(0.3))                             # regularization
        layers.append(ReLU())                                   # activation
        layers.append(Linear(self.n_hidden1, self.n_hidden2))   # first hidden layer
        layers.append(Dropout(0.01))                            # regularization
        layers.append(ReLU())                                   # activation
        layers.append(Linear(self.n_hidden2, self.n_output))    # second hidden layer
        layers.append(Softmax())                                # output layer
        return layers

    def get_loss_func(self):
        """
        Returns the loss function to be used
        """
        return CrossEntropy()

class NeuroseModel(Model):
    """
    The same model that is used in

    https://github.com/irenenikk/neurose
    """

    def __init__(self):
        """
        Construct a new NeuoreModel

        n_inputs - number of inputs
        n_hidden1 - number of nodes in the first hidden layer
        n_hidden2 - number of nodes in the second hidden layer
        n_output - number of outputs
        """
        # model parameters
        self.n_input = 28*28
        self.n_hidden1 = 256
        self.n_hidden2 = 120
        self.n_hidden3 = 64
        self.n_output = 10

    def get_layers(self):
        """
        Returns a list of layers that make this model
        """
        layers = []
        layers.append(ReLU())                                   # activation
        layers.append(Linear(self.n_input, self.n_hidden1))     # input layers
        layers.append(ReLU())                                   # activation
        layers.append(Linear(self.n_hidden1, self.n_hidden2))   # first hidden layer
        layers.append(ReLU())                                   # activation
        layers.append(Linear(self.n_hidden2, self.n_hidden3))   # second hidden layer
        layers.append(ReLU())                                   # activation
        layers.append(Linear(self.n_hidden3, self.n_output))    # second hidden layer
        layers.append(Softmax())                                # output layer
        return layers

    def get_loss_func(self):
        """
        Returns the loss function to be used
        """
        return CrossEntropy()
