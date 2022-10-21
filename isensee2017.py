import tensorflow.keras.backend as K
import numpy as np
import scipy.stats as stats

from functools import partial

from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, BatchNormalization, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization

from tensorflow import pad


class ReflectionPadding3D(Layer):
    """
    3D Reflection Padding
    Attributes:
    - padding: (padding_width, padding_height) tuple
    """
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3] + 2 * self.padding[2], input_shape[4])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height, padding_depth = self.padding
        return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width],\
             [padding_width, padding_width], [0,0]], 'REFLECT')


def create_localization_module(input_layer, n_filters):
    layer1 = ReflectionPadding3D()(input_layer)
    convolution1 = create_convolution_block(layer1, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    layer1 = ReflectionPadding3D()(up_sample)
    convolution = create_convolution_block(layer1, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_last"):
    layer1 = ReflectionPadding3D()(input_layer)
    convolution1 = create_convolution_block(input_layer=layer1, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    layer2 = ReflectionPadding3D()(dropout)
    convolution2 = create_convolution_block(input_layer=layer2, n_filters=n_level_filters)
    return convolution2


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='valid', strides=(1, 1, 1), instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    print(layer.shape)
    if batch_normalization:
        layer = BatchNormalization()(layer)
    elif instance_normalization:
        layer = InstanceNormalization()(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


# it is notable that large swathes of a 21cm co-eval cube are zeros, while will
# impact the values of these statistics. However, we should not cut out the zeros
# since the prediction produced by the network is unlikely to match the number of zero
# voxels.
# It may be of interest to compute a fifth statistic, which would either be the true
# neutral fraction of the co-eval cube, or the number of zero voxels.
def stats_chi2(y_true, y_pred):
    """
    Compute statistical moments for true and predicted data.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_stats = [stats.moment(y_true_f, 1),
        stats.moment(y_true_f, 2),
        stats.moment(y_true_f, 3),
        stats.moment(y_true_f, 4)]
    pred_stats = [stats.moment(y_pred_f, 1),
        stats.moment(y_pred_f, 2),
        stats.moment(y_pred_f, 2),
        stats.moment(y_pred_f, 4)]
    return true_stats, pred_stats


def stats_chi2_loss(y_true, y_pred):
    """
    Compute loss from Chi2 of statistical moments
    """
    true_stats, pred_stats = stats_chi2(y_true, y_pred)
    return np.sum(((pred_stats - true_stats) / true_stats) ** 2)


create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def isensee2017_model(inputs, n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=1, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=stats_chi2_loss, activation_name="sigmoid"):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf
    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf
    :param inputs:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            layer = ReflectionPadding3D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters)
        else:
            layer = ReflectionPadding3D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = Concatenate()([level_output_layers[level_number], up_sampling])
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model


if __name__ == "__main__":
    import tensorflow as tf
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_img = Input((64, 64, 64, 1), name='img')
        model = isensee2017_model(input_img, depth=5, n_segmentation_levels=3)
    model.summary()
