import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from plot import *
from utils import *
from isensee2017 import *
from tqdm import tqdm, tnrange
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import array_to_img, img_to_array, load_img


im_width = int(sys.argv[1])
im_height = int(sys.argv[2])
im_depth = int(sys.argv[3])
lrf = float(sys.argv[4]) # default should be 0.985
batch_size = int(sys.argv[5])
directory = sys.argv[6]
epochs = int(sys.argv[7])
levels = int(sys.argv[8])
seg_levels = int(sys.argv[9])

# load in images

# ids = next(os.walk("scratch/unet3d/masks3d"))[2]  # list of names all images in the given path
ids = ['delta_T_v3_z008.40_nf0.635610_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.574888_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.544232_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.633964_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.656169_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.521659_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.617345_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.637188_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.594391_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.597185_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.654867_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.522856_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.637337_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.635299_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.617298_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.544017_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.544668_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.653014_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.596711_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.617719_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.653178_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.634339_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.615563_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.524477_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.595780_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.520478_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.615448_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.656035_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.594309_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.511529_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.542406_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.594607_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.616766_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.596745_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.519016_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.636751_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.522496_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.521138_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.521185_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.655863_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.636739_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.597140_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.544209_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.655459_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.547098_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.614223_useTs0_200_300Mpc', 'delta_T_v3_z008.40_nf0.636887_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.654219_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.617428_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.545873_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.545418_useTs0_200_300Mpc', 'delta_T_v3_z008.20_nf0.593328_useTs0_200_300Mpc', 'delta_T_v3_z008.30_nf0.614770_useTs0_200_300Mpc', 'delta_T_v3_z008.50_nf0.655648_useTs0_200_300Mpc', 'delta_T_v3_z008.10_nf0.544216_useTs0_200_300Mpc', 'delta_T_v3_z007.85_nf0.473681_useTs0_200_300Mpc', 'delta_T_v3_z008.00_nf0.521530_useTs0_200_300Mpc']
print("No. of images = ", len(ids))

X = np.zeros((len(ids), im_height, im_width, im_depth, 1), dtype=np.float32)
y = np.zeros((len(ids), im_height, im_width, im_depth, 1), dtype=np.float32)

# tqdm is used to display the progress bar
for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    # Load images
    x_img = load_binary_data("scratch/unet3d/images3d/sweep-10-" + id_)
    x_img = shape_data(200, x_img)
    x_img = tf.expand_dims(x_img, 3)
    x_img = resize(x_img, (128, 128, 128, 1), mode='constant', preserve_range=True)
    # Load masks
    mask = load_binary_data("scratch/unet3d/masks3d/" + id_)
    mask = shape_data(200, mask)
    mask = tf.expand_dims(mask, 3)
    mask = resize(mask, (128, 128, 128, 1), mode='constant', preserve_range=True)
    # Save images
    X[n] = x_img
    y[n] = mask# == 0 no binarization for stats loss function!

# split training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, shuffle=False)

name_file = directory + "filenames.txt"
with open(name_file, 'w') as namefn:
    namefn.write(str(ids))
    namefn.close()

# define dice loss function
def dice_loss(y_true, y_pred):
    """
    This calculates the dice coefficient (to be used as loss)
    """  
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true ** 2 + y_pred ** 2, axis=-1)

    return 1 - (numerator + 1) / (denominator + 1)

# initialize model
# compile model
input_img = Input((im_height, im_width, im_depth, 1), name='img')
model = isensee2017_model(input_img, depth=levels, n_segmentation_levels=seg_levels)

model_loc = directory + "model-wedge.h5"

# validation
# load the best model
model.load_weights(model_loc)
# evaluate model on validation set
model.evaluate(X_valid, y_valid, verbose=1)
# predict on training, validation, and testing sets
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)
# preds_train2 = np.zeros(np.shape(preds_train))
# for n, pred in enumerate(preds_train):
    # pred = (pred - pred.min())/pred.max()
    # preds_train2[n] = pred
# preds_train = np.copy(preds_train2)

# preds_val2 = np.zeros(np.shape(preds_val))
# for n, pred in enumerate(preds_val):
    # pred = (pred - pred.min())/pred.max()
    # preds_val2[n] = pred
# preds_val = np.copy(preds_val2)
# threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

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
        locx = locx + str(m) + ".png"
        plot_sample(X_train, y_train, preds_train, preds_train_t, ix=m, dim=n, dir='z')
        plt.savefig(locz)
        plt.clf()
        plt.close()


# validation slice loop
for i in range(4):
    n = 40*i
    # x is the LoS direction
    # y and z are transverse directions
    for j in range(len(y_valid)):
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
        locx = locx + str(m) + ".png"
        plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=j, dim=n, dir='z')
        plt.savefig(locz)
        plt.clf()
        plt.close()


# stats
stats(y_valid, preds_val_t, X_valid, directory)

# visualize layers
visualize_conv_layer('conv3d_2', model, X_train, directory)
visualize_conv_layer('conv3d_6', model, X_train, directory)
visualize_conv_layer('conv3d_10', model, X_train, directory)
visualize_conv_layer('conv3d_14', model, X_train, directory)
visualize_conv_layer('conv3d_16', model, X_train, directory)

# get layer histograms
# conv_layer_hist('conv3d_2', model, X_train, directory)
# conv_layer_hist('conv3d_6', model, X_train, directory)
# conv_layer_hist('conv3d_10', model, X_train, directory)
# conv_layer_hist('conv3d_14', model, X_train, directory)
# conv_layer_hist('conv3d_16', model, X_train, directory)

# save predictions for future use
save_outputs(preds_train, preds_val, ids, directory)