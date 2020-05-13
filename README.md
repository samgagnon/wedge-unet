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

## Files
* **config.py** - Contains the settings on variables relevant to the rest of the code. This should be the only file you need to edit to run the code.

* **model.py** - The definition of the network model.

* **train.py** - Use this to train the network from scratch or to continue training.

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
