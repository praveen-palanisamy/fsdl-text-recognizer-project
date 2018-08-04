import pathlib
from typing import Tuple

from boltons.cacheutils import cachedproperty
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, Reshape, TimeDistributed, Lambda, ZeroPadding2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window


def line_cnn_sliding_window(
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        window_width: float=16,
        window_stride: float=10) -> KerasModel:
    """
    Input is an image with shape (image_height, image_width)
    Output is of shape (output_length, num_classes)
    """
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    image_input = Input(shape=input_shape)
    # (image_height, image_width)

    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    # (image_height, image_width, 1)

    image_patches = Lambda(
        slide_window,
        arguments={'window_width': window_width, 'window_stride': window_stride}
    )(image_reshaped)
    # (num_windows, image_height, window_width, 1)

    # Make a LeNet and get rid of the last two layers (softmax and dropout)
    convnet = lenet((image_height, window_width, 1), (num_classes,))
    convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)

    convnet_outputs = TimeDistributed(convnet)(image_patches)
    # (num_windows, 64)

    # Now we have to get to (output_length, num_classes) shape. One way to do it is to do another sliding window with
    # width = floor(num_windows / output_length)
    # Note that this will likely produce too many items in the output sequence, so take only output_length,
    # and watch out that width is at least 2 (else we will only be able to predict on the first half of the line)

    ##### Your code below (Lab 2)
    # (mum_windows, 64) --> (output_length , num_classes)
    num_windows = int((image_width - window_width) / window_stride) + 1
    width = np.floor(num_windows / output_length)
    slide = 1
    filter_size = num_windows - (output_length -1) *  slide
    slider_output = Conv2D(num_classes, kernel_size=(filter_size, filter_size), activation="relu")(convnet_outputs)
    softmax_output = KerasModel(inputs=slider_output, outputs=Dense(num_classes, activation="softmax"))

    ##### Your code above (Lab 2)

    model = KerasModel(inputs=image_input, outputs=softmax_output)
    model.summary()
    return model

