#!/usr/bin/env python
"""
Trains a neural neutwork with MNIST data using TwoLayerModel
"""

import argparse

from mnist import dataloader
from mnist.networks import NeuralNet
from mnist.optimizers import SGD
from mnist.models import TwoLayerModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int, help='Number of training round (10)')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate (0.1)')
    parser.add_argument('--batch_size', default=256, type=int, help='Mini-batch size (256)')
    parser.add_argument('--load',
                        dest='load_weights',
                        action='store_true',
                        help='Load saved weights from file')

    parser.add_argument('--save',
                        dest='save_weights',
                        action='store_true',
                        help='Save weights to file after training')

    parser.add_argument('--weights_fname',
                        default='../data/saved_weights.dat',
                        help='Path and filename for saving and loading the weights (../data/saved_weights.dat)')

    args = parser.parse_args()

    weights_fname = args.weights_fname

    # create the model
    model = TwoLayerModel()
    net = NeuralNet(model=model)

    if args.load_weights:
        print('- Loading weights from:', weights_fname)
        net.load_weights(weights_fname)

    # create the optimizer
    optimizer = SGD(net=net, dataloader=dataloader.DataLoader(), batch_size=args.batch_size)

    # fit the model
    print('- Training model for', args.epochs, 'epoch, with learning rate', args.lr)
    optimizer.fit(n_epochs=args.epochs, learning_rate=args.lr)

    if args.save_weights:
        print("- Saving weights to:", weights_fname)
        net.save_weights(weights_fname)

