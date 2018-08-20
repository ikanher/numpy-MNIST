import unittest
import numpy as np

from taivasnet.dataloaders import MNISTDataLoader

__author__ = 'Aki Rehn'
__project__ = 'taivasnet'

class TestMNISTDataLoader(unittest.TestCase):
    """
    Test MNIST data loading functionality
    """

    def setUp(self):
        self.loader = MNISTDataLoader()
        self.epsilon = 1e-6

    def test_load_data(self):
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = self.loader.load_data()
        self.assertEqual(x_train.shape, (50000, 784))
        self.assertEqual(y_train.shape, (50000,))
        self.assertEqual(x_valid.shape, (10000, 784))
        self.assertEqual(y_valid.shape, (10000,))
        self.assertEqual(x_test.shape, (10000, 784))
        self.assertEqual(y_test.shape, (10000,))

    def test_normalize(self):
        ((x_train, y_train), (x_valid, y_valid), _) = self.loader.load_data()
        ((x_train, y_train), (x_valid, y_valid)) = \
                self.loader.normalize(((x_train, y_train), (x_valid, y_valid)))

        self.assertTrue(np.isclose(x_train.mean(), 0, atol=self.epsilon))
        self.assertTrue(np.isclose(x_train.std(), 1, atol=self.epsilon))
