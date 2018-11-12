#!/usr/bin/env python3
"""MNIST Example Code."""

# Standard imports
import sys
import time
from datetime import timedelta
import math

# Keras imports
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam

# PIP imports
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Our custom imports
from mnist import MNIST


class KerasCNN(object):
    """Support vector machine class."""

    # Convolutional Layer 1.
    filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
    num_filters1 = 16         # There are 16 of these filters.

    # Convolutional Layer 2.
    filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
    num_filters2 = 36         # There are 36 of these filters.

    # Fully-connected layer.
    fc_size = 128             # Number of neurons in fully-connected laye

    # Get data from files
    data = MNIST(data_dir='/tmp/data/MNIST/')

    # The number of pixels in each dimension of an image.
    img_size = data.img_size

    # The images are stored in one-dimensional arrays of this length.
    img_size_flat = data.img_size_flat

    # Tuple with height and width of images used to reshape arrays.
    img_shape = data.img_shape

    # Tuple with height, width and depth used to reshape arrays.
    # This is used for reshaping in Keras.
    img_shape_full = data.img_shape_full

    # Number of classes, one class for each of 10 digits.
    num_classes = data.num_classes

    # Number of colour channels for the images: 1 channel for gray-scale.
    num_channels = data.num_channels

    def __init__(self):
        """Instantiate the class.

        Args:
            train_batch_size: Training batch size

        Returns:
            None

        """
        # Initialize variables
        fill = 50

        """
        print('{0: <{1}} {2}'.format('Encoded X image:', fill, self.x_image))
        """

        # Start construction of the Keras Sequential model.
        self.model = Sequential()

        # Add an input layer which is similar to a feed_dict in TensorFlow.
        # Note that the input-shape must be a tuple containing the image-size.
        self.model.add(InputLayer(input_shape=(self.img_size_flat,)))

        # The input is a flattened array with 784 elements,
        # but the convolutional layers expect images with shape (28, 28, 1)
        self.model.add(Reshape(self.img_shape_full))

        # First convolutional layer with ReLU-activation and max-pooling.
        self.model.add(
            Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                   activation='relu', name='layer_conv1'))
        self.model.add(MaxPooling2D(pool_size=2, strides=2))

        # Second convolutional layer with ReLU-activation and max-pooling.
        self.model.add(
            Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                   activation='relu', name='layer_conv2'))
        self.model.add(MaxPooling2D(pool_size=2, strides=2))

        # Flatten the 4-rank output of the convolutional layers
        # to 2-rank that can be input to a fully-connected / dense layer.
        self.model.add(Flatten())

        # First fully-connected / dense layer with ReLU-activation.
        self.model.add(Dense(128, activation='relu'))

        # Last fully-connected / dense layer with softmax-activation
        # for use in classification.
        self.model.add(Dense(self.num_classes, activation='softmax'))

        # Model Compilation

        '''
        The Neural Network has now been defined and must be finalized by adding
        a loss-function, optimizer and performance metrics. This is called
        model "compilation" in Keras.

        We can either define the optimizer using a string, or if we want more
        control of its parameters then we need to instantiate an object. For
        example, we can set the learning-rate.
        '''

        optimizer = Adam(lr=1e-3)

        '''
        For a classification-problem such as MNIST which has 10 possible
        classes, we need to use the loss-function called
        categorical_crossentropy. The performance metric we are interested in
        is the classification accuracy.
        '''

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        # Training

        '''
        Now that the model has been fully defined with loss-function and
        optimizer, we can train it. This function takes numpy-arrays and
        performs the given number of training epochs using the given
        batch-size. An epoch is one full use of the entire training-set. So for
        10 epochs we would iterate randomly over the entire training-set 10
        times.
        '''

        self.model.fit(x=self.data.x_train,
                       y=self.data.y_train,
                       epochs=1, batch_size=128)

        # Evaluation

        '''
        Now that the model has been trained we can test its performance on the
        test-set. This also uses numpy-arrays as input.
        '''

        result = self.model.evaluate(x=self.data.x_test, y=self.data.y_test)

        '''
        We can print all the performance metrics for the test-set.
        '''

        for name, value in zip(self.model.metrics_names, result):
            print('{} {}'.format(name, value))

    def plot_example_errors(self, cls_pred):
        """Plot 9 images in a 3x3 grid.

        Function used to plot 9 images in a 3x3 grid, and writing the true and
        predicted classes below each image.

        Args:
            cls_pred: Array of the predicted class-number for all images in the
                test-set.

        Returns:
            None

        """
        # Boolean array whether the predicted class is incorrect.
        incorrect = (cls_pred != self.data.y_test_cls)

        # Get the images from the test-set that have been
        # incorrectly classified.
        images = self.data.x_test[incorrect]

        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = self.data.y_test_cls[incorrect]

        # Plot the first 9 images.
        plot_images(
            images[0:9], self.img_shape, cls_true[0:9], cls_pred=cls_pred[0:9])


def plot_images(images, image_shape, cls_true, cls_pred=None):
    """Plot 9 images in a 3x3 grid.

    Function used to plot 9 images in a 3x3 grid, and writing the true and
    predicted classes below each image.

    Args:
        images: List of images
        image_shape: Shape of each image
        cls_true: List of actual classes associated with each image
        cls_pred: List of predicted classes associated with each image

    Returns:
        None

    """
    # Initialize key variables
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for image_count, axes_object in enumerate(axes.flat):
        # Plot image.
        axes_object.imshow(
            images[image_count].reshape(image_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = 'True: {0}'.format(cls_true[image_count])
        else:
            xlabel = (
                'True: {0}, Pred: {1}'.format(
                    cls_true[image_count], cls_pred[image_count]))

        axes_object.set_xlabel(xlabel)

        # Remove ticks from the plot.
        axes_object.set_xticks([])
        axes_object.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def main():
    """Run main function."""
    # Initialize key variables
    cnn = KerasCNN()

    # Get the first images from the test-set.
    images = cnn.data.x_test[0:9]

    # Get the true classes for those images.
    cls_true = cnn.data.y_test_cls[0:9]

    # Plot the images and labels using our helper-function above.
    plot_images(images, cnn.img_shape, cls_true)

    # Prediction

    '''
    We can also predict the classification for new images. We will just use
    some images from the test-set but you could load your own images into numpy
    arrays and use those instead.
    '''

    images = cnn.data.x_test[0:9]

    '''
    These are the true class-number for those images. This is only used when
    plotting the images.
    '''

    cls_true = cnn.data.y_test_cls[0:9]

    '''
    Get the predicted classes as One-Hot encoded arrays.
    '''

    y_pred = cnn.model.predict(x=images)

    '''
    Get the predicted classes as integers.
    '''

    cls_pred = np.argmax(y_pred, axis=1)
    plot_images(images, cnn.img_shape, cls_true, cls_pred=cls_pred)

    # Examples of Mis-Classified Images

    '''
    We can plot some examples of mis-classified images from the test-set.

    First we get the predicted classes for all the images in the test-set:
    '''

    y_pred = cnn.model.predict(x=cnn.data.x_test)

    '''
    Then we convert the predicted class-numbers from One-Hot encoded arrays to
    integers.
    '''

    cls_pred = np.argmax(y_pred, axis=1)

    '''
    Plot some of the mis-classified images.
    '''

    cnn.plot_example_errors(cls_pred)


if __name__ == "__main__":
    main()
