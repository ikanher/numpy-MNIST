import dataloader
import unittest
import numpy as np

from layers import Softmax, Linear

class TestLinear(unittest.TestCase):
    """
    Test linear layer functionality
    """

    def setUp(self):
        # initialize linear layer with 10 inputs and 3 outputs
        self.linear = Linear(10, 3)

        # test values from pytorch implementation
        self.linear.weights = np.array([
            [-0.1679,  0.0930, -0.0913, -0.0347, -0.3040, -0.1508,  0.1716, -0.0769,  0.3150,  0.2535],
            [-0.0148, -0.2111,  0.1926,  0.0981, -0.2044,  0.2054,  0.1920, 0.2805, -0.1773, -0.0521],
            [-0.0061,  0.0462, -0.2400, -0.2244,  0.1720, -0.0742,  0.1545, 0.0180,  0.1038,  0.0695]])

        self.linear.bias = np.array([ 0.1150,  0.1568, -0.2929])

    def test_forward(self):

        # test values from pytorch implementation
        inputs = np.array([
            [-1.5256, -0.7502, -0.6540, -1.6095, -0.1002, -0.6092, -0.9798, -1.6091, -0.7121,  0.3037],
            [-0.7773, -0.2515, -0.2223,  1.6871, -0.3206, -0.2993,  1.8793, -0.0721,  0.1578, -0.7735],
            [ 0.1991,  0.0457, -1.3924,  2.6891, -0.1110,  0.2927, -0.1578, -0.0288,  2.3571, -1.0373]])

        correct = np.array([
            [ 0.3474, -0.5798, -0.0053],
            [ 0.5082,  0.7010, -0.4063],
            [ 0.5640, -0.1794, -0.4543]])

        result = self.linear.forward(inputs)
        self.assertTrue(np.allclose(result, correct, atol=1e-3))

    def test_backward(self):
        # TBD
        pass


class TestSoftmax(unittest.TestCase):
    """
    Test Softmax functionality
    """

    def setUp(self):
        self.softmax = Softmax()

    def test_forward1(self):
        correct = np.array([[0.26894142, 0.73105858]])
        data = np.array([[1., 2.]])
        result = self.softmax.forward(data)
        self.assertTrue(np.allclose(result, correct))

    def test_forward2(self):
        correct = np.array([[0.84203357, 0.04192238, 0.00208719, 0.11395685]])
        data = np.array([[5., 2., -1, 3]])
        result = self.softmax.forward(data)
        self.assertTrue(np.allclose(result, correct))

    def test_backward(self):
        # TBD
        pass
