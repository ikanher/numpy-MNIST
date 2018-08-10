# Tira project report - Week 3

## What did I do?
* Continued translating PyTorch code to plain numpy
* Implemented backpropagation for simple model with no hidden layers
* Implemented testing of backward pass of layers using gradient checking
* Implemented Dropout layer for regularization
* Implemented ReLU (rectified linear unit) layer that is needed for non-linearity when adding hidden layer(s)
* Added hidden layer to PyTorch implementation
* **Last minute update** I found a bug in my Numpy notebook shuffling code, now finally getting results very similar to PyTorch notebook. Yay!

## How has the program progressed?
Finally got backpropagation working! Also managed to implement tests for backward pass, using gradient checking.

After adding Dropout to the input layer for regularization started to get validation accuracy past 85%, so I'm feeling pretty good about the progress.

## What did I learn?
* How to implement ReLU, Dropout
* How to do gradient checking for backpropagation
* More math and Python, as usually

## What problems did I have?
* Writing gradient checking was way more complicated than I guessed
* I still don't understand why my results are not as good as with the PyTorch implementation, is it doing something behing the scenes?
* Without any regularization I'm overfitting pretty bad (especially if using smaller mini-batches), which is weird as the PyTorch implementation does not do that
* I'm not sure if Dropout should do something special in backprop, some source I've found say yes and some say no (which would make sense, since dropout kills some of the inputs and they _shouldn't_ affect backpropagation)

## What next?
* Try to investigate why PyTorch implementation is getting better results even the code should be equivalent other than not using the framework
* Add a hidden layer to the plain numpy implementation
* If I get the hidden layer implementation working, start moving Neural Network and Gradient Descent code from notebook into their own modules

## Hours
* About 30

