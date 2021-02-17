#  21cm Wedge UNet

A UNet for recovering 21cm intensity information lost to "The Wedge".

## Dependencies
* **python** version 3.5 or more recent
* **numpy**
* **scipy**
* **matplotlib**
* **tensorflow**
* **tqdm**
* **scikit-image**
* **scikit-learn**

## Folders
* **2d_unet** - Contains the 2D version of this code, as well as a file called ``img_maker.py``. The files should be kept in the same directory as the ``/images/`` and ``/masks/`` folders. These folders should contain 2D slices of the images and their associated masks, named from ``0.png`` to ``n.png``. ``img_maker.py`` should be kept in a directory called ``/src/`` which is parallel to the data directory ``/data/``, which should be organized as specified at the end of the readme. The 2D version of the U-Net is no longer being maintained, and depends on keras.

## Files
* **config.py** - Contains the settings on variables relevant to the rest of the code. This should be the only file you need to edit to run the code.

* **isensee2017.py** - The definition of the 3D network model, based on that presented in Isensee (2017).

* **isensee_train.py** - Use this to train the 3D network from scratch or to continue training.

* **isensee_validation.py**  - Use this to load an existing model and run validation on new data.

* **utils.py** - Miscellaneous utilities used in the code. 

* **concaveSitk.py** - Defines watershed segmentation used in plot.py. This code was written by Yin Lin at the University of Chicago.

* **plot.py** - Various utilities used in the visualization of the network outputs.

* **crossCorrelation.py** - Edit the file locations within the file to tell it which data to cross-correlate, then run the script to generate a cross-correlation plot.

## Training the Network

Simply executing ```isensee_train.py``` with appropriate keyword arguments will cause the network to train on whatever images are kept in the ```images/``` and ```masks/``` directories. For example,

```python isensee_train.py 128 128 128 0.985 3 area1/ 100 5 3```

will train a network on images with dimensions of 128x128x128 with a learn rate decay factor of 0.985, a batch size of 3, it will save all outputs in a directory ```./area1/```, the training will run for up to 100 epochs, and the network will be built with 5 levels of depth and three 3 segmentation layers for deep supervision.
