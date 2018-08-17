import unittest
import numpy as np
import os

from taivasnet.networks import NeuralNet
from taivasnet.layers import Linear, Softmax
from taivasnet.losses import CrossEntropy

__author__ = 'Aki Rehn'
__project__ = 'taivasnet'

class TestModelOne():
    """
    Really simple model for testing
    """

    def get_layers(self):
        """
        Returns a list of layers that make this model
        """
        n_input = 50
        n_output = 10
        layers = []
        layers.append(Linear(n_input, n_output))
        layers.append(Softmax())
        return layers

    def get_loss_func(self):
        """
        Returns the loss function to be used
        """
        return CrossEntropy()

class TestModelTwo():
    """
    Another really simple model for testing, with different shape
    """

    def get_layers(self):
        """
        Returns a list of layers that make this model
        """
        n_input = 70
        n_output = 10
        layers = []
        layers.append(Linear(n_input, n_output))
        layers.append(Softmax())
        return layers

    def get_loss_func(self):
        """
        Returns the loss function to be used
        """
        return CrossEntropy()

class TestNeuralNet(unittest.TestCase):
    """
    Test NeuralNet functionality

    Tests only saving and loading weights
    """

    def setUp(self):
        self.weights_fname = '../data/__test_weights.dat'
        self.model1 = TestModelOne()
        self.model2 = TestModelTwo()

    def test_save_and_load_weights(self):
        # create a net with random weights
        net = NeuralNet(model=self.model1)
        w = net.layers[0].weights
        b = net.layers[0].bias

        # save weights
        net.save_weights(self.weights_fname)

        # create a new net with random weights
        net = NeuralNet(model=self.model1)

        # load saved weights
        net.load_weights(self.weights_fname)
        new_w = net.layers[0].weights
        new_b = net.layers[0].bias

        # check that weights are equal
        self.assertTrue(np.alltrue(w == new_w), msg="Saving and loading weights works")
        self.assertTrue(np.alltrue(b == new_b), msg="Saving and loading biases works")

    def test_save_and_load_weights_fails(self):
        # create a net with random weights
        net = NeuralNet(model=self.model1)
        w = net.layers[0].weights
        b = net.layers[0].bias

        # save weights
        net.save_weights(self.weights_fname)

        # create a new net with random weights
        net = NeuralNet(model=self.model2)

        # load saved weights - this should fail
        self.assertRaises(RuntimeError, net.load_weights, self.weights_fname)

    def tearDown(self):
        os.remove(self.weights_fname)
