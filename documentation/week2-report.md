# Tira project report - Week 2

## What did I do?
* Wrote Cross-Entropy Loss and Softmax without using Pytorch features
* Started translating code to plain numpy
* Started building Python modules to be used in Numpy notebook
* Started using some of these modules in Numpy notebook
* Unit tests for modules

## How has the program progressed?
Simple logistic regression PyTorch implementation seems to work like a dream with everything else hand-coded but Backpropagation - still using PyTorch autograd.

Backpropagation is the hardest part, I think. I haven't yet gotten it working either with rewriting PyTorch implementation by hand, nor in the Numpy implementation. I thought I would've had already implemented it this week, but no. So bit disappointed. But I think the program is still progressing quite nicely, step after step.

Couple times I got the backpropagation _almost_ working, but with a really small learning rate (1e-7), and even then, after enough of epochs the loss started growing again and accuracy never got above 70%. Comparing to the PyTorch autograd implementation, where after tree epochs, using learning rate 1e-1, the _validation_ accuracy is above 90%.

## What did I learn?
* More Python
* More math, matrix calculus - especially about the chain rule that is needed for backpropagation
* Cross-Entropy Loss function, Log likelihood, Softmax
* How to build Python modules
* How to unit-test Python code

## What problems did I have?
* I clearly don't yet understand backpropagation completely, wasn't even able to replace PyTorch autograd with hand-written backpropagation code even in a very simple network - not to mention about plain Numpy
* Using Stochastic Gradient Descent with mini-batches seem to make writing backpropagation code a lot more difficult
* Backpropagation, backpropagation and backpropagation
* Still haven't figured out how to generate code coverage, but that's probably not yet important as the main code is still running in a Jupyter Notebook

## What next?
* Try to implement Gradient Checking for debugging backpropagation
* Figure out how to implement backward pass, aka. backpropagation
* Continue moving code from Jupyter Notebooks to Python modules

## Hours
* About 35, maybe even more

