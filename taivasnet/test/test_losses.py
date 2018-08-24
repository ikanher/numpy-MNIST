import unittest
import numpy as np

from taivasnet.losses import CrossEntropy
from taivasnet.layers import Softmax, Linear
from .gradientchecker import GradientChecker

__author__ = 'Aki Rehn'
__project__ = 'taivasnet'

class TestCrossEntropy(unittest.TestCase):
    """
    Test Cross-Entropy loss
    """

    def setUp(self):
        self.cross_entropy = CrossEntropy()

    def test_loss(self):
        correct = 2.3048

        # values calculated in pytorch
        y_pred = np.array([[ 0.1016,  0.1046,  0.0926,  0.1089,  0.0929,  0.0905,  0.0900, 0.1188,  0.1078,  0.0923],
        [ 0.1073,  0.1041,  0.0920,  0.1079,  0.0933,  0.0969,  0.0881, 0.1102,  0.1073,  0.0929],
        [ 0.1070,  0.1069,  0.0925,  0.1122,  0.0916,  0.0885,  0.0879, 0.1092,  0.1101,  0.0941],
        [ 0.0992,  0.1102,  0.0961,  0.1046,  0.0978,  0.0891,  0.0850, 0.1097,  0.1132,  0.0952],
        [ 0.0999,  0.1067,  0.0934,  0.1104,  0.0927,  0.0929,  0.0846, 0.1120,  0.1111,  0.0964],
        [ 0.1113,  0.1015,  0.0921,  0.1107,  0.0913,  0.0946,  0.0861, 0.1139,  0.1080,  0.0907],
        [ 0.0953,  0.1054,  0.0987,  0.1107,  0.0891,  0.0919,  0.0849, 0.1163,  0.1112,  0.0965],
        [ 0.1086,  0.1074,  0.0886,  0.1074,  0.0905,  0.0934,  0.0845, 0.1143,  0.1092,  0.0960],
        [ 0.0981,  0.1059,  0.0999,  0.1088,  0.0895,  0.0901,  0.0862, 0.1153,  0.1091,  0.0970],
        [ 0.1075,  0.1055,  0.0949,  0.1106,  0.0937,  0.0896,  0.0877, 0.1077,  0.1095,  0.0932]])
        y = np.array([ 5,  0,  4,  1,  9,  2,  1,  3,  1,  4])

        loss = self.cross_entropy.loss(y_pred, y)
        self.assertTrue(np.isclose(loss, correct, rtol=1e-4))

    def test_gradient(self):
        """
        Test backpropagation using gradient checking
        """

        # error tolerance
        epsilon = 1e-12

        n_inputs, n_classes = 20, 30
        inputs = np.random.randn(n_inputs, n_classes)
        targets = np.random.randint(n_classes, size=n_inputs)
        grad_output = 1.0
        softmax = Softmax()

        def f(x):
            predictions = softmax.forward(x)
            return self.cross_entropy.loss(predictions, targets)

        grad_inputs_numerical = GradientChecker.eval_numerical_gradient(f, inputs, verbose=False)

        predictions = softmax.forward(inputs)
        grad_inputs = self.cross_entropy.gradient(predictions, targets, inputs)

        self.assertTrue(np.allclose(grad_inputs, grad_inputs_numerical, rtol=epsilon), msg="CrossEntropy gradient calculated correctly")
