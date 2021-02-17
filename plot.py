import os
import random
import numpy as np
import matplotlib.pyplot as plt
import concaveSitk
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from datetime import datetime
from powerSpectrum import *
import scipy.signal as sig

from tensorflow.keras.models import Model


def plot_sample(X, y, preds, binary_preds, ix=None, dim=0, dir='x'):
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X) - 1)
    
    if dir == 'x':
        X = X[:, dim, :, :]
        y = y[:, dim, :, :]
        preds = preds[:, dim, :, :]
        binary_preds = binary_preds[:, dim, :, :]
    if dir == 'y':
        X = X[:, :, dim, :]
        y = y[:, :, dim, :]
        preds = preds[:, :, dim, :]
        binary_preds = binary_preds[:, :, dim, :]
    if dir == 'z':
        X = X[:, :, :, dim]
        y = y[:, :, :, dim]
        preds = preds[:, :, :, dim]
        binary_preds = binary_preds[:, :, :, dim]
    

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='coolwarm')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.01])
    ax[0].set_title('$\Delta$T with wedge')

    ax[1].imshow(y[ix].squeeze(), cmap='coolwarm')
    ax[1].set_title('Ionized regions without wedge')

    ax[2].imshow(preds[ix].squeeze(), cmap='coolwarm', vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.01])
    ax[2].set_title('Ionized regions predicted')

    ax[3].imshow(binary_preds[ix].squeeze(), cmap='coolwarm', vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.01])
    ax[3].set_title('Ionized regions predicted binary')



def visualize_conv_layer(layer_name, model, X, directory):
    """
    Saves 16 plots showing a slice in the first 16 channels of the layer
    input to the module.
    """
  
    layer = model.get_layer(layer_name)
  
    layer_output = layer.output
 
    intermediate_model = Model(inputs=model.input, outputs=layer_output)
 
    intermediate_prediction = intermediate_model.predict(X[0].reshape(1,128,128,128,1))
  
    row_size = 4
    col_size = 4
  
    img_index = 0
 
    print(np.shape(intermediate_prediction))
  
    fig, ax = plt.subplots(row_size,col_size,figsize=(10,8))
 
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(intermediate_prediction[0, :, 0, :, img_index], cmap='hot')
 
            img_index=img_index+1
    fig.savefig(directory + layer_name + ".png")



def conv_layer_hist(layer_name, model, X, directory):
    """
    Saves 16 plots showing a slice in the first 16 channels of the layer
    input to the module.
    """
  
    layer = model.get_layer(layer_name)
  
    layer_output = layer.output
 
    intermediate_model = Model(inputs=model.input, outputs=layer_output)
 
    intermediate_prediction = intermediate_model.predict(X[0].reshape(1,128,128,128,1))

    shape = intermediate_prediction.shape

    plt.hist(np.sum(intermediate_prediction[0,:,:,:,:], axis=(1,2,3))/(shape[1]*shape[2]*shape[3]), bins=30)
    plt.savefig(directory + layer_name + "hist_yzf.png")
    plt.clf()

    plt.hist(np.sum(intermediate_prediction[0,:,:,:,:], axis=(0,2,3))/(shape[0]*shape[2]*shape[3]), bins=30)
    plt.savefig(directory + layer_name + "hist_xzf.png")
    plt.clf()

    plt.hist(np.sum(intermediate_prediction[0,:,:,:,:], axis=(0,1,3))/(shape[0]*shape[1]*shape[3]), bins=30)
    plt.savefig(directory + layer_name + "hist_xyf.png")
    plt.clf()

    plt.hist(np.sum(intermediate_prediction[0,:,:,:,0], axis=(1,2))/(shape[1]*shape[2]), bins=30)
    plt.savefig(directory + layer_name + "hist_yz.png")
    plt.clf()

    plt.hist(np.sum(intermediate_prediction[0,:,:,:,0], axis=(0,2))/(shape[0]*shape[2]), bins=30)
    plt.savefig(directory + layer_name + "hist_xz_allf.png")
    plt.clf()

    plt.hist(np.sum(intermediate_prediction[0,:,:,:,0], axis=(0,1))/(shape[0]*shape[1]), bins=30)
    plt.savefig(directory + layer_name + "hist_xy.png")
    plt.clf()


def tensor_value_history(model, X, directory, show=False):
    """
    Produced a plot showing the average value of the intermediate outputs throughout the network.
    """

    outputs = [layer.output for layer in model.layers]
  
    intermediate_models = [Model(inputs=model.input, outputs=output) for output in outputs]
 
    intermediate_predictions = [intermediate_models[i].predict(X[0].reshape(1,128,128,128,1)) for i in range(len(intermediate_models))]
 
    prediction_avgs = [pred.mean() for pred in intermediate_predictions]

    layer_names=[layer.name for layer in model.layers]

    print(prediction_avgs)

    x = np.array(list(range(len(prediction_avgs))))

    if show:
        plt.xticks(x, layer_names)
        plt.plot(x, prediction_avgs, 'ok')
        plt.show()
        plt.clf()
    else:
        plt.xticks(x, layer_names)
        plt.plot(x, prediction_avgs, 'ok')
        plt.savefig(directory + "tensor_value_history.png")
        plt.clf()

def bubble_distribution(y_valid, y_pred, directory): 
    y_valid_pixels = concaveSitk.waterSeg(h=0.9, imgSize=300, imgDim=128, BW=y_valid[0,:,:,:,0], distReturn=False, fullyConn=True)
    y_pred_pixels = concaveSitk.waterSeg(h=0.9, imgSize=300, imgDim=128, BW=y_pred[0,:,:,:,0], distReturn=False, fullyConn=True)
    plt.clf()
    plt.hist(y_valid_pixels)
    plt.savefig(directory + "ans_valid.png")
    plt.clf()
    plt.hist(y_pred_pixels)
    plt.savefig(directory + "out_valid.png")
    plt.clf()


def nfrac_accuracy(y_valid, y_pred):
    """
    Given the truth and prediction arrays, outputs the neutral fractions of each box in each array.
    The output is a tuple of lists
    """
    # this is bugged somehow. The reported "truth nf" are nowhere near where they should be

    truth_nf = [np.count_nonzero(y_valid[i,:,:,:,0])/128**3 for i in range(y_valid.shape[0])]
    pred_nf = [np.count_nonzero(y_pred[i,:,:,:,0])/128**3 for i in range(y_pred.shape[0])]

    return truth_nf, pred_nf


def res_frac(y_valid, y_pred):
    res_frac_cubes = [y_valid[i,:,:,:,0] - y_pred[i,:,:,:,0] for i in range(y_valid.shape[0])]
    res_frac_array = np.asarray(res_frac_cubes)
    return res_frac_array


# don't use this, use Joelle's code (powerSpectrum)
def pspec(cube):
    """
    Creates power spectrum for given cube.
    Returns two arrays, the distances and the values.
    """
    space_ps = np.abs(np.fft.fftn(cube))
    space_ps *= space_ps

    space_ac = np.fft.ifftn(space_ps).real.round()
    space_ac /= space_ac[0, 0, 0]
    dist = np.minimum(np.arange(200), np.arange(200, 0, -1))
    dist *= dist
    dist_3d = np.sqrt(dist[:, None, None] + dist[:, None] + dist)
    distances, _ = np.unique(dist_3d, return_inverse=True)
    values = np.bincount(_, weights=space_ac.ravel()) / np.bincount(_)

    return distances[1:], values[1:]


def res_stats(y_valid, y_pred):
    """
    Determines the percent of false positives, missed ionized regions, and
    correctly guessed regions for a given residual map.
    """
    FP = [(y_valid[i,:,:,:,0] == 0) * (y_pred[i,:,:,:,0] == 1) for i in range(y_valid.shape[0])]
    TP = [(y_valid[i,:,:,:,0] == 1) * (y_pred[i,:,:,:,0] == 1) for i in range(y_valid.shape[0])]
    FN = [(y_valid[i,:,:,:,0] == 1) * (y_pred[i,:,:,:,0] == 0) for i in range(y_valid.shape[0])]
    TN = [(y_valid[i,:,:,:,0] == 0) * (y_pred[i,:,:,:,0] == 0) for i in range(y_valid.shape[0])]
    FP = [np.count_nonzero(fp) for fp in FP]
    TP = [np.count_nonzero(tp) for tp in TP]
    FN = [np.count_nonzero(fn) for fn in FN]
    TN = [np.count_nonzero(tn) for tn in TN]
    precision = [TP[i]/(TP[i] + FP[i]) for i in range(len(TP))]
    recall = [TP[i]/(TP[i] + FN[i]) for i in range(len(TP))]
    accuracy = [(TP[i] + TN[i])/(TP[i] + FP[i] + TN[i] + FN[i]) for i in range(len(TP))]
    F1 = [(2*precision[i]*recall[i])/(recall[i] + precision[i]) for i in range(len(TP))]
    return precision, recall, accuracy, F1


def stats(y_valid, y_pred, X_valid, directory):
    """
    Produces diagnostics data for network performance.
    """
    # neutral fraction
    truth_nf, pred_nf = nfrac_accuracy(y_valid, y_pred)
    line1 = "True Neutral Fractions" + str(truth_nf) + "\n"
    line2 = "Predicted Neutral Fractions:" + str(pred_nf)
    line = line1 + line2
    fname = directory + "truth_v_prednf.txt"
    with open(fname, 'w') as fn:
        fn.write(line)
        fn.close()
    # residual map
    map_array = res_frac(y_valid, y_pred)
    reslock = directory + "residual_map.png"
    plt.imsave(reslock, map_array[0,0])
    plt.clf()
    # power spectra
    spec_loc = directory + "power_spectra.png"
    k_mask, power_mask = PowerSpectrum(y_valid[0,:,:,:,0]).compute_pspec()
    k_pred, power_pred = PowerSpectrum(y_pred[0,:,:,:,0]).compute_pspec()
    k_img, power_img = PowerSpectrum(X_valid[0,:,:,:,0]).compute_pspec()
    kpar_mask_cyl, kperp_mask_cyl, power_mask_cyl = PowerSpectrum(y_valid[0,:,:,:,0]).compute_cylindrical_pspec(return_bins=True)
    kpar_pred_cyl, kperp_pred_cyl, power_pred_cyl = PowerSpectrum(y_pred[0,:,:,:,0]).compute_cylindrical_pspec(return_bins=True)
    kpar_img_cyl, kperp_img_cyl, power_img_cyl = PowerSpectrum(X_valid[0,:,:,:,0]).compute_cylindrical_pspec(return_bins=True)
    fig, axs = plt.subplots(2, 3)
    axs[0,0].plot(k_mask, power_mask)
    axs[0,1].plot(k_pred, power_pred)
    axs[0,2].plot(k_img, power_img)
    axs[1,0].imshow(y_valid[0,0,:,:,0])
    axs[1,1].imshow(y_pred[0,0,:,:,0])
    axs[1,2].imshow(X_valid[0,0,:,:,0])
    plt.savefig(spec_loc)
    plt.clf()
    cyl_loc = directory + "cylindrical_spectra.png"
    fig, axs = plt.subplots(2, 3)
    axs[0,0].imshow(np.log(power_mask_cyl, out=np.zeros_like(power_mask_cyl), where=(power_mask_cyl!=0.0)), origin="lower")
    axs[0,1].imshow(np.log(power_pred_cyl, out=np.zeros_like(power_pred_cyl), where=(power_pred_cyl!=0.0)), origin="lower")
    axs[0,2].imshow(np.log(power_img_cyl, out=np.zeros_like(power_img_cyl), where=(power_img_cyl!=0.0)), origin="lower")
    axs[1,0].imshow(y_valid[0,0,:,:,0])
    axs[1,1].imshow(y_pred[0,0,:,:,0])
    axs[1,2].imshow(X_valid[0,0,:,:,0])
    plt.savefig(cyl_loc)
    plt.clf()
    # residual map stats
    precision, recall, accuracy, F1 = res_stats(y_valid, y_pred)
    statfn = directory + "map_stats.txt"
    with open(statfn, 'w') as fn:
        line = ""
        for i in range(len(precision)):
            line += "Map " + str(i) + "\n"
            line += "Precision " + str(precision[i]) + "\n"
            line += "Recall " + str(recall[i]) + "\n"
            line += "Accuracy " + str(accuracy[i]) + "\n"
            line += "F1 " + str(F1[i]) + "\n"
        fn.write(line)
        fn.close()
    stat_array_loc = directory + "map_stats.csv"
    res_stat_array = np.array([precision, recall, accuracy, F1])
    np.savetxt(stat_array_loc, res_stat_array, delimiter=',')
    # pspec cross-correlation
    croscorloc = directory + "cross_correlation.png"
    cordata = sig.correlate(power_pred, power_mask, 'same')
    plt.plot(k_mask, cordata, '-k')
    plt.savefig(croscorloc)
    plt.clf()


def save_outputs(preds_train, preds_val, ids, directory):
    for i in range(preds_val.shape[0]):
        # save validation predictions
        preds_val_name = directory + "val-predict-" + ids[preds_train.shape[0] + i]
        preds_val[i,:,:,:,0].tofile(preds_val_name)
    # for i in range(preds_train.shape[0]):
        # save training predictions
        # preds_train_name = directory + "train-predict-" + ids[i]
        # preds_train[i,:,:,:,0].tofile(preds_train_name)