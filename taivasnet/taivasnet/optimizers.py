"""
Module contains optimizers for training a Neural Netowrk
"""

__author__ = 'Aki Rehn'
__project__ = 'taivasnet'

import numpy as np

class SGD():
    """
    Stochastic Gradient Descent with mini-batches
    """

    def __init__(self, net=None, dataloader=None, batch_size=256, use_all_valid=True, shuffle=True):
        """
        Constructs an SGD optimizer

        net - NeuralNet model to be used
        dataloader - DataLoader object for loading training data
        batch_size - Mini-batch size
        use_all_valid - If true will use whole validation set for calculating validaiton
                        accuracy, otherwise will only use batch_size random samples.
        shuffle - If true the training data will be shuffled
        """

        self.net = net
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.use_all_valid = use_all_valid
        self.shuffle = shuffle
        self._load_data()

    def _load_data(self):
        # load the training data using DataLoader object
        ((x_train, y_train), (x_valid, y_valid), _) = self.dataloader.load_data()

        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def _shuffle(self, arr1, arr2):
        """
        Shuffles arr1 and arr2 in the same order
        """
        random_idxs = np.arange(len(arr1))
        np.random.shuffle(random_idxs)
        return arr1[random_idxs], arr2[random_idxs]

    def fit(self, n_epochs=5, learning_rate=1e-2, suppress_output=False):
        """
        Trains the model

        n_epochs - Number of epochs to train
        learning_rate - The learning rate to use
        """

        if not suppress_output:
            fmt = '{:<5} {:12} {:12} {:6} {:6}'
            print(fmt.format('Epoch', 'Train loss', 'Valid loss', 'Train acc', 'Valid acc'))

        for epoch in range(n_epochs):

            for i in range(0, len(self.x_train), self.batch_size):

                inputs = self.x_train[i:i+self.batch_size]
                targets = self.y_train[i:i+self.batch_size]

                if self.shuffle:
                    inputs, targets = self._shuffle(inputs, targets)

                # forward propagation
                y_pred = self.net.forward(inputs)
                predictions = y_pred.copy()

                # calculate the loss
                loss = self.net.loss(predictions, targets)
                grad_loss = self.net.loss_gradient(predictions, targets)

                # backpropagation
                self.net.backward(grad_loss)

                # update weights
                for layer in self.net.layers:
                    if layer.learning:
                        layer.weights -= learning_rate * layer.grad_w
                        layer.bias -= learning_rate * layer.grad_b

            # calculate validation loss
            self.net.train = False
            if self.use_all_valid:
                # use whole validation set
                y_valid_pred = self.net.forward(self.x_valid)
                loss_valid = self.net.loss(y_valid_pred, self.y_valid)
            else:
                # pick batch_size random items for calculating validation accuracy
                random_idxs = np.random.randint(0, len(x_valid), batch_size)
                y_valid_pred = self.net.forward(x_valid[random_idxs])
                loss_valid = self.net.loss(y_valid_pred, y_valid[random_idxs])
            self.net.train = True

            # calculate accuracy and validation accuracy
            accuracy = np.mean(y_pred.argmax(axis=1) == targets)
            valid_accuracy = np.mean(y_valid_pred.argmax(axis=1) == self.y_valid)

            if not suppress_output:
                fmt = '{:<5} {:03.10f} {:03.10f} {:02.7f} {:02.7f}'
                print(fmt.format(epoch, loss, loss_valid, accuracy, valid_accuracy))
