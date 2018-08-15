import unittest
import numpy as np

from mnist.layers import Softmax, Linear, ReLU, Dropout
from mnist.losses import CrossEntropy

__author__ = 'Aki Rehn'
__project__ = 'mnist'

class TestLinear(unittest.TestCase):
    """
    Test linear layer functionality
    """

    def setUp(self):
        # initialize linear layer with 10 inputs and 3 outputs
        self.linear = Linear(10, 3)

    def test_forward(self):

        # test values from pytorch implementation
        self.linear.weights = np.array([
            [-0.1679,  0.0930, -0.0913, -0.0347, -0.3040, -0.1508,  0.1716, -0.0769,  0.3150,  0.2535],
            [-0.0148, -0.2111,  0.1926,  0.0981, -0.2044,  0.2054,  0.1920, 0.2805, -0.1773, -0.0521],
            [-0.0061,  0.0462, -0.2400, -0.2244,  0.1720, -0.0742,  0.1545, 0.0180,  0.1038,  0.0695]]).T

        self.linear.bias = np.array([ 0.1150,  0.1568, -0.2929])

        inputs = np.array([
            [-1.5256, -0.7502, -0.6540, -1.6095, -0.1002, -0.6092, -0.9798, -1.6091, -0.7121,  0.3037],
            [-0.7773, -0.2515, -0.2223,  1.6871, -0.3206, -0.2993,  1.8793, -0.0721,  0.1578, -0.7735],
            [ 0.1991,  0.0457, -1.3924,  2.6891, -0.1110,  0.2927, -0.1578, -0.0288,  2.3571, -1.0373]])

        correct = np.array([
            [ 0.3474, -0.5798, -0.0053],
            [ 0.5082,  0.7010, -0.4063],
            [ 0.5640, -0.1794, -0.4543]])

        result = self.linear.forward(inputs)
        self.assertTrue(np.allclose(result, correct, atol=1e-3), msg="Linear layer forward pass has errors")

    def test_backward(self):
        """
        Test backpropagation using gradient checking
        """

        # error tolerance
        epsilon = 1e-12

        inputs = np.random.randn(10, 6)
        weights = np.random.randn(6, 5)
        bias = np.random.randn(5)
        grad_output = np.random.randn(10, 5)
        linear = Linear(60, 5)

        linear.weights, linear.bias = weights, bias

        grad_inputs_numerical = eval_numerical_gradient_array(
                lambda x: linear.forward(inputs), inputs, grad_output)

        grad_weights_numerical = eval_numerical_gradient_array(
                lambda x: linear.forward(inputs), weights, grad_output)

        grad_bias_numerical = eval_numerical_gradient_array(
                lambda x: linear.forward(inputs), bias, grad_output)

        grad_inputs, grad_weights, grad_bias = linear.backward(grad_output)

        self.assertTrue(np.allclose(grad_inputs, grad_inputs_numerical, rtol=epsilon), msg="Linear grad_inputs has errors")
        self.assertTrue(np.allclose(grad_weights, grad_weights_numerical, rtol=epsilon), msg="Linear grad_weights has errors")
        self.assertTrue(np.allclose(grad_bias, grad_bias_numerical, rtol=epsilon), msg="Linear grad_bias has errors")

class TestSoftmax(unittest.TestCase):
    """
    Test Softmax functionality
    """

    def setUp(self):
        self.softmax = Softmax()
        self.cross_entropy = CrossEntropy()

    def test_forward1(self):
        correct = np.array([[0.26894142, 0.73105858]])
        data = np.array([[1., 2.]])
        result = self.softmax.forward(data)
        self.assertTrue(np.allclose(result, correct), msg="Softmax forward1 has errors")

    def test_forward2(self):
        correct = np.array([[0.84203357, 0.04192238, 0.00208719, 0.11395685]])
        data = np.array([[5., 2., -1, 3]])
        result = self.softmax.forward(data)
        self.assertTrue(np.allclose(result, correct), msg="Softmax forward2 has errors")

    def test_backward(self):
        """
        Test backpropagation using gradient checking
        """

        # error tolerance
        epsilon = 1e-12

        n_inputs, n_classes = 20, 30
        inputs = np.random.randn(n_inputs, n_classes)
        targets = np.random.randint(n_classes, size=n_inputs)
        grad_output = 1.0

        def f(x):
            predictions = self.softmax.forward(x)
            return self.cross_entropy.loss(predictions, targets)

        grad_inputs_numerical = eval_numerical_gradient(f, inputs, verbose=False)

        predictions = self.softmax.forward(inputs)
        grad_inputs = self.softmax.backward(targets)

        self.assertTrue(np.allclose(grad_inputs, grad_inputs_numerical, rtol=epsilon), msg="Softmax backward has errors")

class TestReLU(unittest.TestCase):
    """
    Test ReLU functionality
    """

    def setUp(self):
        self.relu = ReLU()

    def test_forward1(self):
        data = np.random.randn(5, 5)

        # make sure at least one negative value exists
        data[0, 0] = -5

        output = self.relu.forward(data)
        self.assertFalse(output[output < 0].any(), msg="ReLU forward1 has errors")

    def test_forward2(self):
        data = np.random.randn(10, 10)
        output = self.relu.forward(data)
        self.assertFalse(output[output < 0].any(), msg="ReLU forward2 has errors")

    def test_backward(self):
        """
        Test backpropagation using gradient checking
        """

        # error tolerance
        epsilon = 1e-12

        x = np.random.randn(3, 2, 8, 8)
        grad_output = np.random.randn(3, 2, 8, 8)

        grad_numerical = eval_numerical_gradient_array(lambda x: self.relu.forward(x), x, grad_output)

        output = self.relu.forward(x)
        grad = self.relu.backward(grad_output)

        self.assertTrue(np.allclose(grad, grad_numerical, rtol=epsilon), msg="ReLU backward has errors")

class TestDropout(unittest.TestCase):
    """
    Test Dropout layer functionality
    """

    def test_forward1(self):

        # probability of dropping inputs
        p = 0.7

        data = np.random.randn(10, 50)
        dropout = Dropout(p=p)
        output = dropout.forward(data)

        zeros = output[output == 0]
        pct = zeros.size / data.size

        self.assertTrue(np.isclose(pct, p, rtol=1e-1), msg="Dropout forward1 has errors")

    def test_forward2(self):

        # probability of dropping inputs
        p = 0.2

        data = np.random.randn(40, 30)
        dropout = Dropout(p=0.2)
        output = dropout.forward(data)

        zeros = output[output == 0]
        pct = zeros.size / data.size

        self.assertTrue(np.isclose(pct, p, rtol=1e-1), msg="Dropout forward2 has errors")


"""
Gradient checking code from:
cs231n.github.io/assignments2017/assignment1
"""

def eval_numerical_gradient(f, x, verbose=True, h=1e-5):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext() # step to next dimension

    return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
