import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from datetime import datetime


def plot_sample(X, y, preds, binary_preds, ftype, ix=None):
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='inferno')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.01])
    ax[0].set_title('$\Delta$T with wedge')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Ionized regions without wedge')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.01])
    ax[2].set_title('Ionized regions predicted')

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.01])
    ax[3].set_title('Ionized regions predicted binary')

    time_str = datetime.now().time().strftime('%H:%M')
    fname = ftype + time_str
    plt.savefig(fname)
