import os
import random
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

from utils import *
from config import *
from model import *
from plot import *
from tqdm import tqdm, tnrange
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

ids = next(os.walk("scratch/unet3d/masks3d"))[2]  # list of names all images in the given path
print("No. of images = ", len(ids))

X = np.zeros((len(ids), im_height, im_width, im_depth, 1), dtype=np.float32)
y = np.zeros((len(ids), im_height, im_width, im_depth, 1), dtype=np.float32)

# tqdm is used to display the progress bar
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    # Load images
    x_img = load_binary_data("scratch/unet3d/images/sweep-10-" + id_)
    x_img = shape_data(200, x_img)
    x_img = tf.expand_dims(x_img, 3)
    x_img = resize(x_img, (im_height, im_width, 1), mode='constant', preserve_range=True)
    # Load masks
    mask = load_binary_data("scratch/unet3d/masks/" + id_)
    mask = shape_data(200, mask)
    mask = tf.expand_dims(mask, 3)
    mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)
    # Save images
    X[n] = x_img / 255.0
    y[n] = mask / 255.0 == 0

# split training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, shuffle=False)

# compile model
input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=n_filters, dropout=0.05, 
batchnorm=False)
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

model_loc = directory + "model-wedge.h5"

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    ModelCheckpoint(model_loc, verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit(X_train, y_train, batch_size=2, epochs=100, 
callbacks=callbacks, validation_data=(X_valid, y_valid))

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
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

train0_loc = directory + "train0.png"

plot_sample(X_train, y_train, preds_train, preds_train_t, ix=0, dim=0)
plt.savefig(train0_loc)
plt.clf()
plt.close()

train75_loc = directory + "train75.png"

plot_sample(X_train, y_train, preds_train, preds_train_t, ix=0, dim=75)
plt.savefig(train75_loc)
plt.clf()
plt.close()

train02_loc = directory + "train0-2.png"

plot_sample(X_train, y_train, preds_train, preds_train_t, ix=2, dim=0)
plt.savefig(train02_loc)
plt.clf()
plt.close()

train752_loc = directory + "train75-2.png"

plot_sample(X_train, y_train, preds_train, preds_train_t, ix=2, dim=75)
plt.savefig(train752_loc)
plt.clf()
plt.close()

val0_loc = directory + "val0.png"

plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=0, dim=0)
plt.savefig(val0_loc)
plt.clf()
plt.close()

val75_loc = directory + "val75.png"

plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=0, dim=75)
plt.savefig(val75_loc)
plt.clf()
plt.close()
