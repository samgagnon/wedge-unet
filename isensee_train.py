"""
Author: Samuel Gagnon-Hartman

This network will seek to replicate that in https://arxiv.org/pdf/1802.10508.pdf
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import generator

plt.style.use("ggplot")

from utils import *
from isensee2017 import *
from plot import *
from tqdm import tqdm
from skimage.io import imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

im_width = int(sys.argv[1])
im_height = int(sys.argv[2])
im_depth = int(sys.argv[3])
lrf = float(sys.argv[4]) # default should be 0.985
batch_size = int(sys.argv[5])
directory = sys.argv[6]
epochs = int(sys.argv[7])
levels = int(sys.argv[8])
seg_levels = int(sys.argv[9])

# ids = ['delta_T_v3_z008.40_nf0.635610_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.574888_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.544232_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.633964_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.656169_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.521659_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.617345_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.637188_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.594391_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.597185_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.654867_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.522856_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.637337_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.635299_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.617298_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.544017_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.544668_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.653014_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.596711_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.617719_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.653178_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.634339_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.615563_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.524477_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.595780_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.520478_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.615448_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.656035_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.594309_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.511529_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.542406_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.594607_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.616766_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.596745_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.519016_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.636751_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.522496_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.521138_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.521185_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.655863_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.636739_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.597140_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.544209_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.655459_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.547098_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.614223_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.636887_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.654219_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.617428_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.545873_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.545418_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.593328_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.614770_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.655648_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.544216_useTs0_200_300Mpc', 'delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.521530_useTs0_200_300Mpc']
ids = ['delta_T_v3_z008.40_nf0.635610_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.574888_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.635610_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.574888_useTs0_200_300Mpc']
print("No. of images = ", len(ids))

X = np.zeros((len(ids), im_height, im_width, im_depth, 1), dtype=np.float32)
y = np.zeros((len(ids), im_height, im_width, im_depth, 1), dtype=np.float32)

# tqdm is used to display the progress bar
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    # Load images
    # x_img = load_binary_data("../../GitHub/wedge-unet/bar2/bar-2-" + id_, np.float64)
    x_img = load_binary_data("images3d/sweep-10-" + id_, np.float32)
    x_img = x_img.reshape((128,128,128,1))
    # x_img = tf.expand_dims(x_img, 3)
    x_img = x_img - x_img.min()
    x_img = x_img / x_img.max()
    x_img = resize(x_img, (im_height, im_width, im_depth, 1), mode='symmetric', preserve_range=True)
    # Load masks
    mask = load_binary_data("masks3d/" + id_, dtype=np.float32)
    # mask = load_binary_data("../../GitHub/wedge-unet/masks3d/" + id_)
    mask = mask.reshape((128,128,128,1))
    # mask = tf.expand_dims(mask, 3)
    mask = resize(mask, (im_height, im_width, im_depth, 1), mode='symmetric', preserve_range=True)
    # Save images
    X[n] = x_img # add the /255 if necessary
    y[n] = mask #< 0.9 no binarization of masks for stats loss function!

print("ALL FILES LOADED")

# split training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.50, shuffle=False)

tf.debugging.set_log_device_placement(False)

model_loc = directory + "model-wedge.h5"

def schedule(epoch, lr):
    """
    Learning rate scheduler.
    """
    new_rate = 5e-4 * (lrf ** epoch)
    return new_rate

# define callbacks
callbacks = [
    EarlyStopping(patience=20, verbose=1),
    LearningRateScheduler(schedule, verbose=1),
    ModelCheckpoint(model_loc, verbose=1, save_best_only=True, save_weights_only=True)
]


input_img = Input((im_height, im_width, im_depth, 1), name='img')
model = isensee2017_model(input_img, depth=levels, n_segmentation_levels=seg_levels)
model.summary()
results = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(X_valid, y_valid))

model_loc = directory + "model-wedge.h5"

loss_loc = directory + "loss.csv"
val_loc = directory + "val_loss.csv"

np.savetxt(loss_loc, results.history["loss"], delimiter=',')
np.savetxt(val_loc, results.history["val_loss"], delimiter=',')

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()
plot_loc = directory + "loss_plot.png"
plt.savefig(plot_loc)
plt.clf()
plt.close()

# validation

# load the best model
model.load_weights(model_loc)
# evaluate model on validation set
model.evaluate(X_valid, y_valid, verbose=1)
# predict on training, validation, and testing sets
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)
# change cutoff
preds_train2 = np.zeros(np.shape(preds_train))
for n, pred in enumerate(preds_train):
    pred = (pred - pred.min())/pred.max()
    preds_train2[n] = pred
preds_train = np.copy(preds_train2)

preds_val2 = np.zeros(np.shape(preds_val))
for n, pred in enumerate(preds_val):
    pred = (pred - pred.min())/pred.max()
    preds_val2[n] = pred
preds_val = np.copy(preds_val2)
# threshold predictions
preds_train_t = (preds_train > 0.9).astype(np.uint8)
preds_val_t = (preds_val > 0.9).astype(np.uint8)


# bubble size distribution
# bubble_distribution(y_valid, preds_val_t, directory)


# training slice loop
for i in range(4):
    n = 40*i
    # x is the LoS direction
    # y and z are transverse directions
    for j in range(4):
        locx = directory + "train" + str(n) + "x-"
        locy = directory + "train" + str(n) + "y-"
        locz = directory + "train" + str(n) + "z-"
        m = 4*j
        locx = locx + str(m) + ".png"
        plot_sample(X_train, y_train, preds_train, preds_train_t, ix=m, dim=n, dir='x')
        plt.savefig(locx)
        plt.clf()
        plt.close()
        locy = locy + str(m) + ".png"
        plot_sample(X_train, y_train, preds_train, preds_train_t, ix=m, dim=n, dir='y')
        plt.savefig(locy)
        plt.clf()
        plt.close()
        locz = locx + str(m) + ".png"
        plot_sample(X_train, y_train, preds_train, preds_train_t, ix=m, dim=n, dir='z')
        plt.savefig(locz)
        plt.clf()
        plt.close()


# validation slice loop
for i in range(4):
    n = 40*i
    # x is the LoS direction
    # y and z are transverse directions
    for j in range(4):
        locx = directory + "val" + str(n) + "x-"
        locy = directory + "val" + str(n) + "y-"
        locz = directory + "val" + str(n) + "z-"
        m = j
        locx = locx + str(m) + ".png"
        plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=j, dim=n, dir='x')
        plt.savefig(locx)
        plt.clf()
        plt.close()
        locy = locy + str(m) + ".png"
        plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=j, dim=n, dir='y')
        plt.savefig(locy)
        plt.clf()
        plt.close()
        locz = locx + str(m) + ".png"
        plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=j, dim=n, dir='z')
        plt.savefig(locz)
        plt.clf()
        plt.close()

# # stats
# stats(y_valid, preds_val_t, X_valid, directory)

# # tensor value history
# tensor_value_history(model, X_train, directory)

# visualize_conv_layer('conv3d_2', model, X_train, directory)
# visualize_conv_layer('conv3d_4', model, X_train, directory)
# visualize_conv_layer('conv3d_6', model, X_train, directory)
# visualize_conv_layer('conv3d_10', model, X_train, directory)
# visualize_conv_layer('conv3d_15', model, X_train, directory)
# visualize_conv_layer('conv3d_19', model, X_train, directory)
# visualize_conv_layer('conv3d_25', model, X_train, directory)

# save predictions for future use
save_outputs(preds_train, preds_val, ids, directory)
