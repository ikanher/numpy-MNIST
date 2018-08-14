"""
Contains different dataloaders
"""

__author__ = 'Aki Rehn'
__project__ = 'mnist'

import pickle
import gzip

class DataLoader():
    """
    Loads the mnist training, validation and testing data.

    Data from http://deeplearning.net/data/mnist/mnist.pkl.gz--2018-07-25
    """

    def __init__(self, mnist_path='../data'):
        self.mnist_path = mnist_path
        self.mnist_fname = 'mnist.pkl.gz'

    def _load_mnist(self):
        mnist_data_file = self.mnist_path + '/' + self.mnist_fname
        with gzip.open(mnist_data_file, 'rb') as f:
            return pickle.load(f, encoding='latin-1')

    def normalize(self, data):
        """
        Normalizes training and validation data.
        """
        (x_train, y_train), (x_valid, y_valid) = data

        # calculate mean and standard deviation
        mean = x_train.mean()
        std = x_train.std()

        # normalize training data
        x_train = (x_train - mean) / std

        # normalize validation data
        x_valid = (x_valid-mean)/std

        return ((x_train, y_train), (x_valid, y_valid))


    def load_data(self):
        """
        Loads mnist data.

        Returns training, validation and test sets.
        """
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = self._load_mnist()
        return ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))
