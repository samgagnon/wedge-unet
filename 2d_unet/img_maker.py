import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from os.path import basename
from config import *


# loads binary data and stores in numpy array
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


# reshapes data into cube
def shape_data(dimension, data, sky=False):
    if sky:
        data.shape = (82, 127, 127)
        data = data.reshape((82, 127, 127), order='F')
        if data.max() != 0:
            data = data/data.max()
        return data
    else:
        data.shape = (dimension, dimension, dimension)
        data = data.reshape((dimension, dimension, dimension), order='F')
        if data.max() != 0:
            data = data/data.max()
        return data


# selects one in fifty slices from the cube to stack
def slice_selector(x, sky=False):
    """
    Converts input cube of type np.ndarray into a list of numpy arrays representing
    necessary (uncorrelated) cube slices
    """
    # the number of maps is four times the number of map cubes passed into the method
    # this is at the current sampling rate of four maps per cube
    if sky:
        cube_slices = [x[i, :, :] for i in range(0, 82)]
        #cube_slices2 = [x[:, i, :29] for i in range(0, 127)]
        #cube_slices3 = [x[:, :29, i] for i in range(0, 127)]
        #cube_slices = cube_slices2 + cube_slices3
        return cube_slices
    else:
        cube_slices1 = [x[i, :, :] for i in range(0, 200)]
        cube_slices2 = [x[:, i, :] for i in range(0, 200)]
        cube_slices3 = [x[:, :, i] for i in range(0, 200)]
        cube_slices = cube_slices1 + cube_slices2 + cube_slices3
        return cube_slices

# forces maps to take values of zero or 1
def monochrome(x):
    """
    Takes input list of float64 np arrays and outputs list of int np arrays
    """
    print("SHAPE OF X:",np.shape(x))
    print("TYPE OF X:", type(x))
    monochrome_list = []
    for map in x:
        rounded_array = np.round(x,0)
        monochrome_list.append(rounded_array)
    
    print("SHAPE OF MONOCHROME:",np.shape(monochrome_list))
    print("TYPE OF MONOCHROME:", type(monochrome_list))
    print(monochrome_list[0][0])
    return monochrome_list[0]

# generates two lists of 2D maps
def generate_maps(prefix='sweep-10', training_data_folder='../data/', verbose=True):
    if verbose:
        print("Starting to load files into DataFeeder...")
        print("Checking Folder:" + training_data_folder)

    xH = []
    T = []
    xH_maps_counter = 0
    delta_T_maps_counter = 0
    img1 = 0
    img2 = 0

    target = 'delta_T'
    target_name_length = len(target)

    if prefix is None:
        train_filename = 'delta_T'
        train_name_length = len(train_filename)
        sky = False
    elif prefix == 'sweep-10-sky' or prefix == 'sweep-10-conv':
        sky = True
    else:
        train_filename = prefix + '-delta_T'
        train_name_length = len(prefix) + 8
        sky = False

    for directory in os.listdir(training_data_folder):
        if basename(directory)[0] == 'z' and basename(directory)[2] != '1' and basename(directory)[2] != '7':
        # if basename(directory)[0] == 'z': # includes very low and very high redshifts
            if verbose:
                print(directory)
            file_folder = training_data_folder + directory
            for folder in os.listdir(file_folder):
                if basename(folder)[0:3] == 'Run':
                    if verbose:
                        print(folder)
                    file_subfolder = file_folder + '/' + folder
                    for filename in os.listdir(file_subfolder):
                        # loads neutral hydrogen maps
                        if basename(filename)[0:target_name_length] == target and xH_maps_counter < xH_maps_max:
                            img_filename = prefix + '-' + filename
                            if sky:
                                mask_prefix = prefix.split("-")[-1]
                                mask_filename = mask_prefix + "-" + filename
                            else:
                                mask_filename = filename
                            if verbose:
                                print(mask_filename)
                                print(prefix + '-' + mask_filename)
                            mask_data = load_binary_data(file_subfolder + '/' + mask_filename)
                            if training_data_folder=='../michael_data/':
                                mask_DIM = int("" + mask_filename.split("_")[-3])
                            else:
                                mask_DIM = int("" + mask_filename.split("_")[-2])
                            mask_data = shape_data(mask_DIM, mask_data, sky)
                            mask_data = slice_selector(mask_data, sky)
                            # xH = xH + data
                            for slice in mask_data:
                                slice = slice - slice.min()
                                plt.imsave('../unet_images/masks/' + str(img1) + '.png', slice, cmap='gray')
                                img1 += 1
                            xH_maps_counter += 1

                            img_data = load_binary_data(file_subfolder + '/' + img_filename)
                            if training_data_folder=="../michael_data/":
                                img_DIM = int("" + img_filename.split("_")[-3])
                            else:    
                                img_DIM = int("" + img_filename.split("_")[-2])
                            img_data = shape_data(img_DIM, img_data, sky)
                            img_data = slice_selector(img_data, sky)
                            # xH = xH + data
                            for slice in img_data:
                                slice = slice - slice.min()
                                plt.imsave('../unet_images/imgs/' + str(img2) + '.png', slice, cmap='gray')
                                img2 += 1
                                message = "Maps saved:" + str(img2)
                                print(message)
                            delta_T_maps_counter += 1

    # T = [t - t.min() for t in T]
    # xH = [x - x.min() for x in xH]
    # print("length of source and target",len(T),len(xH))
    # print(T[0].max(), T[0].min())
    # print(type(T), type(T[0]))
    # return T, xH


# saves all images in list to unet_images
def image_saver(T, xH):
    print("Now saving images")
    t_counter = 0
    xh_counter = 0
    # xH = monochrome(xH)
    for image in T:
        plt.imsave('../unet_images/imgs/' + str(t_counter) + '.png', image, cmap='gray')
        t_counter+=1
    for image in xH:
        plt.imsave('../unet_images/masks/' + str(xh_counter) + '.png', image, cmap='gray')
        xh_counter+=1


print("Executing file")
# T, xH = generate_maps()
# image_saver(T, xH)
generate_maps(prefix='sweep-10', training_data_folder='../data/')
