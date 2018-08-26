"""
Contains different dataloaders
"""

__author__ = 'Aki Rehn'
__project__ = 'taivasnet'

import numpy as np
import pickle
import gzip

class MNISTDataLoader():
    """
    Loads the MNIST training, validation and testing data.

    Data from http://deeplearning.net/data/mnist/mnist.pkl.gz--2018-07-25
    """

    def __init__(self, mnist_path='../data', n_samples=None):
        """
        Constructor

        mnist_path - path to mnist.pkl.gz
        n_samples - how many samples to use (default all)
        """
        self.mnist_path = mnist_path
        self.mnist_fname = 'mnist.pkl.gz'
        self.n_samples = n_samples

    def _load_mnist(self):
        mnist_data_file = self.mnist_path + '/' + self.mnist_fname

        with gzip.open(mnist_data_file, 'rb') as f:
            ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding='latin-1')
        if self.n_samples != None:
            x_train = x_train[:self.n_samples]
            y_train = y_train[:self.n_samples]

        # normalize training data
        (x_train, y_train), (x_valid, y_valid) = \
                self._normalize(((x_train, y_train), (x_valid, y_valid)))

        return ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))

    def _normalize(self, data):
        """
        Normalizes training and validation data.
        """
        (x_train, y_train), (x_valid, y_valid) = data

        # calculate mean and standard deviation
        mean = x_train.mean()
        std = x_train.std()

        # normalize training data
        x_train = (x_train - mean)/std

        # normalize validation data
        x_valid = (x_valid-mean)/std

        return ((x_train, y_train), (x_valid, y_valid))

    def load_data(self):
        """
        Loads MNIST data.

        Returns training, validation and test sets.
        """
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = self._load_mnist()
        return ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))

class RegressionDataLoader():

    def load_data(self):
        """
        Generates some data for linear regression, y = 2x

        Returns training, validation and test sets.
        """
        randoms = np.random.randint(1, 1000, 100)
        x = np.array([[x] for x in randoms])
        y = np.multiply(x, 2)

        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = \
                ((x[:60], y[:60]), (x[60:80], y[60:80]), (x[80:], y[80:]))

        return ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))
