import os
import random
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
<<<<<<< HEAD
<<<<<<< HEAD
=======
%matplotlib inline
>>>>>>> 51a18f3723d5c46b2e4fd1eb8ded0efd68993eb5
=======
>>>>>>> cce32fa6336d918808a0e960df9c3207479866bc

from utils import *
from config import *
from model import *
from tqdm import tqdm, tnrange
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

ids = next(os.walk("images"))[2] # list of names all images in the given path
ids = sorted_nicely(ids)
print("No. of images = ", len(ids))

X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

# tqdm is used to display the progress bar
for n, id_ in notebook.tqdm(enumerate(ids), total=len(ids)):
    # Load images
    x_img = load_binary_data("images3d/sweep-10-"+id_)
    x_img = shape_data(200, x_img)
    x_img = tf.expand_dims(x_img, 3)
    x_img = resize(x_img, (128, 128, 128, 1), mode = 'constant', preserve_range = True)
    # Load masks
    mask = load_binary_data("masks3d/"+id_)
    mask = shape_data(200, mask)
    mask = tf.expand_dims(mask, 3)
    mask = resize(mask, (128, 128, 128, 1), mode = 'constant', preserve_range = True)
    # Save images
    X[n] = x_img/255.0
    y[n] = mask/255.0==0

# split training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, shuffle=False)

# compile model
input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-wedge.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit(X_train, y_train, batch_size=32, epochs=50, callbacks=callbacks,\
                    validation_data=(X_valid, y_valid))

# validation
# load the best model
model.load_weights('model-wedge.h5')
# evaluate model on validation set
model.evaluate(X_valid, y_valid, verbose=1)
# predict on training, validation, and testing sets
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)
# threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

# save plots
# training
plot_sample(X_train, y_train, preds_train, preds_train_t, ftype='train-', ix=0)
# validation
plot_sample(X_valid, y_valid, preds_val, preds_val_t, ftype='val-', ix=400)
