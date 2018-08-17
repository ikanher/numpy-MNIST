# Implementation

A more general Neural Network was implemented. So it could be used for other datasets than MNIST only.

Code is written in Python and is using [Numpy](http://www.numpy.org/) for matrix operations.

## Code architecture

### Modules

Code is divided into six different Python modules.

* [dataloaders.py](../taivasnet/taivasnet/dataloader.py) - contains classes for loading data, currently only one DataLoader is implemented, taivasnetDataLoader
* [models.py](../taivasnet/taivasnet/models.py) - contains models, which are basically object wrapping list of layers and a loss function
* [layers.py](../taivasnet/taivasnet/layers.py) - contains layers used in models
  * Layer - abstract class that all layers should inherit
  * Linear - Linear layer
  * Softmax - Softmax output layer
  * Dropout - Dropout regularization layer
  * ReLU - Rectified Linear Unit activation layer
* [losses.py](../taivasnet/taivasnet/losses.py) - contains loss functions
  * CrossEntropy - calculates Cross-Entropy loss
* [networks.py](../taivasnet/taivasnet/networks.py) - contains neural network implementations
  * NeuralNet - generic neural network that takes different models as input
* [optimizers.py](../taivasnet/taivasnet/optimizers.py) - contains different optimizers
  * SGD - Stochastic Gradient Descent with mini-batches optimizer

### Class diagram

This diagram gives an overview of how all the objects are connected to each other.

![Class diagram](class-diagram-v1.png)

## Neural network model architecture

Neural network models are defined in [models.py](../taivasnet/taivasnet/models.py). The model that is currently used by [train.py](../taivasnet/train.py) is the only model currently defined, TwoLayerModel.

TwoLayerModel architecture is pretty simple.

![](mlp.png)
*Image from http://corochann.com/mnist-training-with-multi-layer-perceptron-1149.html*

In TwoLayerModel case:

* Input layer has 784 (28x28 images flattened) units
* First hidden layer has 256 units
* Second hidden layer has 256 units
* Output layer has 10 units
* And what's missing from the image the output layer is connected to _Softmax_ layer which finally gives out 10 probabilities for different classes

However, by passing different attributes to the constructor you can easily change the number of units per layer.

Between the layers there are _Dropout_ layers for regularization and _ReLU_ layers for non-linearity.

