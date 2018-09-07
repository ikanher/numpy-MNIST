#!/usr/bin/env python
"""
Trains a neural neutwork with MNIST data using TwoLayerModel
"""

import argparse
import torch
import numpy as np
import time

from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from taivasnet.dataloaders import MNISTDataLoader

class TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
                nn.Dropout(0.01),
                nn.Linear(28*28, 256),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.Dropout(0.01),
                nn.ReLU(),
                nn.Linear(64, 10),
                nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

def _shuffle(arr1, arr2):
    """
    Shuffles arr1 and arr2 in the same order
    """
    random_idxs = np.arange(len(arr1))
    np.random.shuffle(random_idxs)
    return arr1[random_idxs], arr2[random_idxs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int, help='Number of training round (10)')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate (0.1)')
    parser.add_argument('--batch_size', default=256, type=int, help='Mini-batch size (256)')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Enable cuda')

    args = parser.parse_args()
    learning_rate = args.lr
    n_epochs = args.epochs
    batch_size = args.batch_size
    use_cuda = args.cuda

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # create the datasets
    dataloader = MNISTDataLoader()
    (x_train, y_train), (x_valid, y_valid), _ = dataloader.load_data()
    (x_train, y_train), (x_valid, y_valid) = dataloader._normalize(((x_train, y_train), (x_valid, y_valid)))
    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    x_valid = torch.from_numpy(x_valid).to(device)
    y_valid = torch.from_numpy(y_valid).to(device)

    # create the model
    model = TwoLayerModel().to(device)

    # create the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # fit the model
    print('- Training model for', n_epochs, 'epoch, with learning rate', learning_rate)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    fmt = '{:<5} {:12} {:12} {:6} {:6} {}'
    print(fmt.format('Epoch', 'Train loss', 'Valid loss', 'Train acc', 'Valid acc', 'Seconds'))

    train_start = time.time()
    for epoch in range(n_epochs):

        epoch_start = time.time()
        for i in range(0, len(x_train), batch_size):
            inputs = x_train[i:i+batch_size]
            targets = y_train[i:i+batch_size]

            inputs, targets = _shuffle(inputs, targets)

            y_pred = model(inputs)

            loss = loss_fn(y_pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            valid_pred = model(x_valid)
            loss_valid = loss_fn(valid_pred, y_valid)
            model.train()

            if use_cuda:
                accuracy = np.mean(y_pred.detach().cpu().numpy().argmax(axis=1) == targets.detach().cpu().numpy())
                valid_accuracy = np.mean(valid_pred.detach().cpu().numpy().argmax(axis=1) == y_valid.detach().cpu().numpy())
            else:
                accuracy = np.mean(y_pred.detach().numpy().argmax(axis=1) == targets.detach().numpy())
                valid_accuracy = np.mean(valid_pred.detach().numpy().argmax(axis=1) == y_valid.detach().numpy())

            epoch_elapsed = time.time() - epoch_start
            fmt = '{:<5} {:03.10f} {:03.10f} {:02.7f} {:02.7f} {:05.3f}'
            print(fmt.format(epoch, loss, loss_valid, accuracy, valid_accuracy, epoch_elapsed))

    train_elapsed = time.time() - train_start
    print('Training finished, elapsed {:05.3f} seconds.'.format(train_elapsed))
