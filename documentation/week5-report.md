# Tira project report - Week 5

## What did I do?
* Peer review 1
* Started with peer review 2
* Add more tests
* Inspired by peer-reviewed project, moved the responsibility of calculating the loss gradient to the loss functions
* Studied how could Linear Regression with simple MSE loss function could be best integrated into the library

## How has the program progressed?
I was quite happy to manage write unit tests for NeuralNet and leaving only the SGD optimizer to be tested with integration tests.

Managed to partly implemented Linear Regression with MSE for the proejct, but the code is working only with a really, really small learning rate. So there must be some flaw in the implementation and I haven't decided to commit the code yet.

## What did I learn?
* Unit testing
* Linear regression & MSE and it's derivatives

## What problems did I have?
* Implementing Linear regression with MSE support to the library was not as straight-forward as I expected - I could write it easily as separate code, but putting it into the library gives headaches with problems like how to connect the MSE gradient to the Linear layers...

## What next?
* Finalize documentation
* Finish peer review 2
* If I still got time to get it working, try to implement the Linear regression support to the library with a simple demo

## Hours
* Around 30 (including maybe 12 hours for peer review(s))

