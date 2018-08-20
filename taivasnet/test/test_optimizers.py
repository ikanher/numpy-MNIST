import unittest
import numpy as np
import os

from taivasnet.networks import NeuralNet
from taivasnet.models import TwoLayerModel
from taivasnet.optimizers import SGD
from taivasnet.dataloaders import MNISTDataLoader

__author__ = 'Aki Rehn'
__project__ = 'taivasnet'

class TestSGD(unittest.TestCase):
    """
    Test SGD functionality using TwoLayerModel

    Just tests that the loss goes down after fitting the model.
    Uses small subset of the actual MNIST data for testing.
    """

    def setUp(self):
        n_samples = 100
        model = TwoLayerModel()
        self.net = NeuralNet(model=model)
        self.loader = MNISTDataLoader(n_samples=n_samples)
        self.optimizer = SGD(net=self.net, dataloader=self.loader, batch_size=n_samples)

    def test_fit_one_epoch(self):
        ((x_train, y_train), (x_valid, y_valid), _) = self.loader.load_data()

        y_pred_begin = self.net.forward(x_train)
        train_loss_begin = self.net.loss(y_pred_begin, y_train.astype(int))

        self.optimizer.fit(n_epochs=1, learning_rate=1e-1, suppress_output=True)

        y_pred_after = self.net.forward(x_train)
        train_loss_after = self.net.loss(y_pred_after, y_train)

        self.assertTrue(np.less(train_loss_after, train_loss_begin), msg="Loss is getting smaller in the beginning")

    def test_fit_two_epochs(self):
        ((x_train, y_train), (x_valid, y_valid), _) = self.loader.load_data()

        self.optimizer.fit(n_epochs=1, learning_rate=1e-1, suppress_output=True)

        y_pred_begin = self.net.forward(x_train)
        train_loss_begin = self.net.loss(y_pred_begin, y_train.astype(int))

        self.optimizer.fit(n_epochs=1, learning_rate=1e-1, suppress_output=True)

        y_pred_after = self.net.forward(x_train)
        train_loss_after = self.net.loss(y_pred_after, y_train)
        self.assertTrue(np.less(train_loss_after, train_loss_begin), msg="Loss is getting smaller after 1 epochs")

    def test_fit_five_epochs(self):
        ((x_train, y_train), (x_valid, y_valid), _) = self.loader.load_data()

        self.optimizer.fit(n_epochs=4, learning_rate=1e-1, suppress_output=True)

        y_pred_begin = self.net.forward(x_train)
        train_loss_begin = self.net.loss(y_pred_begin, y_train.astype(int))

        self.optimizer.fit(n_epochs=1, learning_rate=1e-1, suppress_output=True)

        y_pred_after = self.net.forward(x_train)
        train_loss_after = self.net.loss(y_pred_after, y_train)
        self.assertTrue(np.less(train_loss_after, train_loss_begin), msg="Loss is getting smaller after 4 epochs")
