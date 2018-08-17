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

**If you already have an earlier version of Anaconda installed** you might have to install numpy and matplotlib separately. You can do it by saying

```
conda install numpy
````

And respectively

```
conda install matplotlib
```

## You are ready!

### Train the network

This will train the network with default values.

```
cd numpy-MNIST
./train.py --save
```

Or alternatively you can add some options.

```
$ ./train.py --epochs 20 --save
- Training model for 20 epoch, with learning rate 0.1
Epoch Train loss   Valid loss   Train acc Valid acc
0     1.5630066073 1.4939244801 0.4625000 0.4545000
1     0.4858543611 0.4722542815 0.8500000 0.8605000
2     0.2826413546 0.2965191751 0.9375000 0.9150000
3     0.1655711033 0.2195714570 0.9500000 0.9367000
4     0.1746750307 0.1758653760 0.9375000 0.9510000
5     0.1120270871 0.1469914880 0.9875000 0.9593000
6     0.1107354093 0.1320762823 0.9750000 0.9619000
7     0.0814344076 0.1144965976 0.9875000 0.9677000
8     0.0715982803 0.1077676272 0.9750000 0.9697000
9     0.0694175710 0.1027864482 0.9750000 0.9701000
10    0.0441153274 0.0937176742 1.0000000 0.9736000
11    0.0692730487 0.0895030952 0.9750000 0.9743000
12    0.0557225957 0.0867564994 0.9750000 0.9750000
13    0.0473420582 0.0843610814 1.0000000 0.9762000
14    0.0632664255 0.0815441377 0.9875000 0.9770000
15    0.0302250556 0.0779607758 1.0000000 0.9782000
16    0.0426277603 0.0753950672 0.9875000 0.9794000
17    0.0270402511 0.0762966677 1.0000000 0.9782000
18    0.0302901822 0.0773504286 0.9875000 0.9779000
19    0.0129810146 0.0712935601 1.0000000 0.9801000
- Saving weights to: ../data/saved_weights.dat
```

For more options, see

```
./train.py -h
```

### Want to see some predictions?

MNIST dataset comes with test images. There is a script that loads random images from the test set and displays them with prediction and actual value.

```
./predict.py
```

Note that that predictions loop will continue for ever and spawn a new image every time you close the image window. To completely exit from the program, use Ctrl-C in the window where you started the script.

