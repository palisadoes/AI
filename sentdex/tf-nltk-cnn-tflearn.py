#!/usr/bin/env python3
"""Script to demonstrate  basic tensorflow machine learning."""

# Standard imports
import os
import sys

# PIP3 imports
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist


def convolutional_neural_network():
    """Create the neural network model.

    Args:
        _data: Data

    Returns:
        output: Output

    """
    # Initialize key variables
    conv1_filter_count = 32
    conv2_filter_count = 64
    fc_units = 1024
    image_height = 28
    image_width = 28
    filter_size = 2
    pooling_kernel_size = 2
    keep_probability = 0.8
    fully_connected_units = 10

    # Create the convolutional network stuff
    convnet = input_data(
        shape=[None, image_height, image_width, 1], name='input')

    convnet = conv_2d(
        convnet, conv1_filter_count, filter_size, activation='relu')
    convnet = max_pool_2d(convnet, pooling_kernel_size)

    convnet = conv_2d(
        convnet, conv2_filter_count, filter_size, activation='relu')
    convnet = max_pool_2d(convnet, pooling_kernel_size)

    convnet = fully_connected(convnet, fc_units, activation='relu')
    convnet = dropout(convnet, keep_probability)

    convnet = fully_connected(
        convnet, fully_connected_units, activation='softmax')
    convnet = regression(
        convnet,
        optimizer='adam',
        learning_rate=0.01,
        loss='categorical_crossentropy',
        name='targets')

    return convnet


def main():
    """Main Function.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    epochs_to_try = 10
    n_classes = 10
    keep_rate = 0.8

    # We sometimes get insufficient memory errors if this is not set low
    desired_test_sample_size = 1000

    # Define the image width and height of images
    width = 28
    height = 28

    # Get the data from mnist
    vectors, classes, test_x, test_y = mnist.load_data(one_hot=True)

    # Reshape the test and live vectors
    vectors = vectors.reshape([-1, width, height, 1])
    test_x = test_x.reshape([-1, width, height, 1])

    model = tflearn.DNN(convolutional_neural_network())
    model.fit(
        {'input': vectors},
        {'targets': classes},
        n_epoch=epochs_to_try,
        validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=500,
        show_metric=True,
        run_id='mnist')

    print(
        np.round(model.predict([test_x[1]])),
        model.predict([test_x[1]])
    )
    print(test_y[1])


if __name__ == "__main__":
    main()
