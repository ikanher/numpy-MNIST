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

class TestModelThree():
    """
    Another really simple model for testing the backward and forward with just two
    Linear layers.
    """

    def get_layers(self):
        """
        Returns a simple model with one hidden layer and softmax output
        """
        n_input = 10
        n_hidden = 6
        n_output = 5
        layers = []
        layers.append(Linear(n_input, n_hidden))
        layers.append(Linear(n_hidden, n_output))
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
        self.model3 = TestModelThree()

    def test_forward(self):
        inputs = np.random.randn(6, 10)

        net = NeuralNet(model=self.model3)

        linear1 = Linear(10, 6)
        linear1.weights = net.layers[0].weights
        linear1.bias = net.layers[0].bias

        linear2 = Linear(6, 5)
        linear2.weights = net.layers[1].weights
        linear2.bias = net.layers[1].bias

        correct = linear1.forward(inputs)
        correct = linear2.forward(correct)

        x = net.forward(inputs)

        self.assertTrue(np.array_equal(x, correct), msg="Net forward pass is the same as forward of the layers combined")

    def test_backward(self):
        inputs = np.random.randn(6, 10)

        net = NeuralNet(model=self.model3)

        linear1 = Linear(10, 6)
        linear1.weights = net.layers[0].weights
        linear1.bias = net.layers[0].bias

        linear2 = Linear(6, 5)
        linear2.weights = net.layers[1].weights
        linear2.bias = net.layers[1].bias

        # first calculate thte forward pass
        fwd = linear1.forward(inputs)
        fwd = linear2.forward(fwd)

        grad_output = np.random.randn(6, 5)

        # now we can calculate the backward pass
        grad_correct, _, _ = linear2.backward(grad_output)
        grad_correct, _, _ = linear1.backward(grad_correct)

        net.forward(inputs)
        grad = net.backward(grad_output)

        self.assertTrue(np.array_equal(grad, grad_correct), msg="Net backward pass is the same as backward of the layers combined")

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

        # remove the temporarily saved weights
        os.remove(self.weights_fname)

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

        # remove the temporarily saved weights
        os.remove(self.weights_fname)
