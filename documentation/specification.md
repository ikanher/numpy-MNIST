# Specification document

## Problem that I'm trying to solve
* How to train a Neural Network using [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) with plain numpy
* How to use the trained Neural Network to classify digits from input images

### Algorithms
* [Stochastic Gradient Descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) for training the network (optimizing the network weights)
* [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) for calculating the gradients that SGD needs for adjusting the weights of the networks

### Data structures
* Tensors (n dimensional matrices) - the weights of the Neural Network

### Input
* In the training phase the program is using [MNIST database in Python pickle format](http://deeplearning.net/data/mnist/mnist.pkl.gz)
* In the classifying phase the program is getting 28x28 pixel grayscale images representing digits as inputs and the program is supposed to output the actual digit the image contains

### Time and space complexity
* No idea - [best reference I found after quick search](https://ai.stackexchange.com/questions/5728/time-complexity-for-training-a-neural-network)

### References
* [Steve Renals - Stochastic gradient descent; classification](http://www.inf.ed.ac.uk/teaching/courses/mlp/2015/mlp02-sln.pdf)
* [Chris Williams - Optimization](http://www.inf.ed.ac.uk/teaching/courses/mlpr/2015/slides/09_optimization.pdf)
* [Justin Johnson - Backpropagation for a Linear Layer](http://cs231n.stanford.edu/handouts/linear-backprop.pdf)
* [ML Cheatsheet - Backpropagation](http://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html)
* [PyTorch documentation](https://pytorch.org/docs/stable/index.html)

