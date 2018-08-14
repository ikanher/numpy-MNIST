# Getting started

If you want to test out the code here are instructions how to easily get all required dependencies installed.

## Install Anaconda

Easiest way to install dependencies is using Anaconda.

So [download Anaconda](https://www.anaconda.com/download/) and install it into your environment following the [installation guides](https://docs.anaconda.com/anaconda/install/).

### Create a new conda environment

```
conda create -n mnist
```

## Clone the repository

```
git clone https://github.com/ikanher/numpy-MNIST.git
```

## Activate conda environment

```
conda activate mnist
```

Or if the above is not working, use:

```
source activate mnist
```

## You are ready!

### Train the network

This will train the network with default values.

```
cd numpy-MNIST
./train.py
```

### Want to see some predictions?

MNIST dataset comes with test images. There is a script that loads random images from the test set and displays them with prediction and actual value.

```
./predict.py
```

Note that that predictions loop will continue for ever and spawn a new image every time you close the image window. To completely exit from the program, use Ctrl-C in the window where you started the script.

