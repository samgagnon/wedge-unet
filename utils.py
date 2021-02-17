import numpy as np
import re
import sys


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def load_binary_data(filename, dtype=np.float32):
    """
    We assume that the data was written
    with write_binary_data() (little endian).
    """
    f = open(filename, "rb")
    data = f.read()
    f.close()
    _data = np.frombuffer(data, dtype)
    if sys.byteorder == 'big':
        _data = _data.byteswap()
    return _data

def shape_data(dimension, data):
    """
    Reshapes 3D data into cube.
    """
    data.shape = (dimension, dimension, dimension)
    data = data.reshape((dimension, dimension, dimension), order='F')
    return data
