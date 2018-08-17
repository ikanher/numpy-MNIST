# Tira project report - Week 4

## What did I do?
* Added support for one hidden layer
* Added support for two hidden layers
* Added support for N hidden layers
* Moved rest of the code from Jupyter notebooks to Python modules and scripts
* Created a more generalised version of a Neural Network that can take different models as arguments and use them for training
* Created simple CLI scripts for training and predicting
* Fixed code structure to support code coverage reports
* Created first versions of testing and implementation documents
* Renamed the python module from 'mnist' to 'taivasnet' to reflect that is is a more general Neural Network library than just specific for MNIST

## How has the program progressed?
Really good! The program is starting to take it's final form now.

## What did I learn?
* Backpropagating through more layers is not as difficult as it sounds when you got the basics correct
* Python code coverage
* About ELU layer and Kaiming normalization for deeper networks, tested this with PyTorch but didn't achieve much better results (adding Convolutional layers would be the most obvious way to improve past 98% accuracy)
* About momentum for SGD, though not did not implement it (this would make training faster)
* About weight decay for SGD, though did not implement it (this might actually improve results a little bit as another method for regularization)

## What problems did I have?
* I'm bit unsure how and what level I should test the NeuralNet and SGD objects
* Else everything has gone quite fine after fixing the shuffle bug last week that caused poorer results compared to the PyTorch models

## What next?
* Figure out testing of SGD and NeuralNet objects (integration tests?)
* Finalize documentation
* Figure out if there is something more that needs to be done
* If I have time make an example of using the network for something else than MNIST that can be trained on a CPU

## Hours
* Around 35

