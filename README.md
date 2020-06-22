#  21cm Wedge UNet

A UNet for recovering 21cm intensity information lost to "The Wedge".

## Dependencies
* **python** version 3.5 or more recent
* **numpy**
* **scipy**
* **matplotlib**
* **keras**
* **tensorflow**
* **tqdm**
* **scikit-image**
* **scikit-learn**

## Folders
* **2d_unet** - Contains the 2D version of this code, as well as a file called ``img_maker.py``. The files should be kept in the same directory as the ``/images/`` and ``/masks/`` folders. These folders should contain 2D slices of the images and their associated masks, named from ``0.png`` to ``n.png``. ``img_maker.py`` should be kept in a directory called ``/src/`` which is parallel to the data directory ``/data/``, which should be organized as specified at the end of the readme.

## Files
* **config.py** - Contains the settings on variables relevant to the rest of the code. This should be the only file you need to edit to run the code.

* **model.py** - The definition of the 3D network model.

* **train.py** - Use this to train the 3D network from scratch or to continue training.

* **model3d.py**  - The definition of the 3D network model.

* **train3d.py** - Use this to train the 3D network from scratch or to continue training.

* **utils.py** - Miscellaneous utilities used in the code. 

## Files in data
* **fourier.py** - Applies wedge effects to data stored in the folder.

* **cleaner.py** - Removes files of a specified extension from database.

## Folders in data
Data files should be organized within the directories below
```
data/redshift/Run x - RNG y/files
```
## Training the Network

Simply executing ```train.py``` will cause the network to train on whatever images are kept in the ```images/``` and ```masks/``` directories.
