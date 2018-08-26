# Usage

## Basic usage

If you just want to do some training and predictions using the predefined model, see [Getting started](getting-started.md).

## Adding your own model

To add your own model just add them to [models.py](../taivasnet/taivasnet/models.py).

In the `get_layers` function, just create a Python list of layers and return those. Predefined layers are defined in [layers.py](../taivasnet/taivasnet/models.py). You can also add new layers if you like. Currently implemented are `Linear`, `Softmax`, `Dropout` and `ReLU`.

The `get_loss_func` should return the loss function to be used. Loss functions are defined in [losses.py](../taivasnet/taivasnet/losses.py). If you need more loss functions, you can add them here.

## Using your own data

For using your own data, implement a new `DataLoader` object. These are defined in [dataloaders.py](../taivasnet/taivasnet/dataloaders.py).

## Training

To train, you can use the [train.py](../taivasnet/train.py) script.

To use your own model here, first import it, and then change the lines that read

```
    # create the model
    model = TwoLayerModel()
```

.. to use your own model.

If you want to use your own dataloader, it should be passed to the `SGD` optimizer, by changing the lines that read

```
    # create the optimizer
    optimizer = SGD(net=net, dataloader=MNISTDataLoader(), batch_size=args.batch_size)
```

Also don't remember to import your own dataloader.

## Predicting

When training the model you can ask it to save the weights to a file

```
./train.py --save
```

Or alternatively you can give your own filename instead of the default.

```
./train.py --save --weights_fname /path/to/file.dat
```

Then in the prediction script, you should load the saved weights and are now ready to make predictions.

There is an example [predict.py](../taivasnet/predict.py) that does predictions for images in the MNIST test set. Take a look in here for example, but note that the way to do predictions does wary depending on which data set you are using.
