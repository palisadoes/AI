#!/usr/bin/env python3
"""MNIST Example Code."""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

from mnist import MNIST


class ConvolutionalNeuralNetwork(object):
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

    # Number of classes, one class for each of 10 digits.
    num_classes = data.num_classes

    # Number of colour channels for the images: 1 channel for gray-scale.
    num_channels = data.num_channels

    def __init__(self):
        """Instantiate the class.

        Args:
            None

        Returns:
            None

        """
        # Setup tensors

        '''
        Placeholder variables serve as the input to the TensorFlow
        computational graph that we may change each time we execute the graph.
        We call this feeding the placeholder variables and it is demonstrated
        further below.

        First we define the placeholder variable for the input images. This
        allows us to change the images that are input to the TensorFlow graph.
        This is a so-called tensor, which just means that it is a
        multi-dimensional vector or matrix. The data-type is set to float32 and
        the shape is set to [None, img_size_flat], where None means that the
        tensor may hold an arbitrary number of images with each image being a
        vector of length img_size_flat.
        '''
        x_vectors = tf.placeholder(tf.float32, shape=[None, self.img_size_flat], name='x')

        '''
        The convolutional layers expect x to be encoded as a 4-dim tensor so we
        have to reshape it so its shape is instead [num_images, img_height,
        img_width, num_channels]. Note that img_height == img_width == img_size
        and num_images can be inferred automatically by using -1 for the size
        of the first dimension. So the reshape operation is:
        '''
        x_image = tf.reshape(
            x_vectors, [-1, self.img_size, self.img_size, self.num_channels])

        '''
        Next we have the placeholder variable for the true labels associated
        with the images that were input in the placeholder variable x. The
        shape of this placeholder variable is [None, num_classes] which means
        it may hold an arbitrary number of labels and each label is a vector of
        length num_classes which is 10 in this case.
        '''
        y_true = tf.placeholder(
            tf.float32, shape=[None, self.num_classes], name='y_true')

        '''
        We could also have a placeholder variable for the class-number, but we
        will instead calculate it using argmax. Note that this is a TensorFlow
        operator so nothing is calculated at this point.
        '''
        y_true_cls = tf.argmax(y_true, axis=1)

        # Convolutional Layer 1

        '''
        Create the first convolutional layer. It takes x_image as input and
        creates num_filters1 different filters, each having width and height
        equal to filter_size1. Finally we wish to down-sample the image so it
        is half the size by using 2x2 max-pooling.
        '''
        layer_conv1, weights_conv1 = new_conv_layer(
            x_image,
            self.num_channels, self.filter_size1, self.num_filters1,
            use_pooling=True)

        # Convolutional Layer 2

        '''
        Create the second convolutional layer, which takes as input the output
        from the first convolutional layer. The number of input channels
        corresponds to the number of filters in the first convolutional layer.
        '''
        layer_conv2, weights_conv2 = new_conv_layer(
            layer_conv1,
            self.num_filters1, self.filter_size2, self.num_filters2,
            use_pooling=True)

        # Flatten layer

        '''
        The convolutional layers output 4-dim tensors. We now wish to use these
        as input in a fully-connected network, which requires for the tensors
        to be reshaped or flattened to 2-dim tensors.
        '''
        layer_flat, num_features = flatten_layer(layer_conv2)

        # Fully-Connected Layer 1

        '''
        Add a fully-connected layer to the network. The input is the flattened
        layer from the previous convolution. The number of neurons or nodes in
        the fully-connected layer is fc_size. ReLU is used so we can learn
        non-linear relations.
        '''
        layer_fc1 = new_fc_layer(
            layer_flat, num_features, self.fc_size, use_relu=True)

        # Fully-Connected Layer 2

        '''
        Add another fully-connected layer that outputs vectors of length
        10 for determining which of the 10 classes the input image belongs to.
        Note that ReLU is not used in this layer.
        '''
        layer_fc2 = new_fc_layer(
            layer_fc1, self.fc_size, self.num_classes, use_relu=False)

        # Predicted Class

        '''
        The second fully-connected layer estimates how likely it is that the
        input image belongs to each of the 10 classes. However, these estimates
        are a bit rough and difficult to interpret because the numbers may be
        very small or large, so we want to normalize them so that each element
        is limited between zero and one and the 10 elements sum to one. This is
        calculated using the so-called softmax function and the result is
        stored in y_pred.

        The class-number is the index of the largest element.
        '''
        y_pred = tf.nn.softmax(layer_fc2)
        y_pred_cls = tf.argmax(y_pred, axis=1)

        # Cost-function to be optimized

        '''
        To make the model better at classifying the input images, we must
        somehow change the variables for all the network layers. To do this
        we first need to know how well the model currently performs by
        comparing the predicted output of the model y_pred to the desired
        output y_true.

        The cross-entropy is a performance measure used in classification. The
        cross-entropy is a continuous function that is always positive and if
        the predicted output of the model exactly matches the desired output
        then the cross-entropy equals zero. The goal of optimization is
        therefore to minimize the cross-entropy so it gets as close to zero
        as possible by changing the variables of the network layers.

        TensorFlow has a built-in function for calculating the cross-entropy.
        Note that the function calculates the softmax internally so we must use
        the output of layer_fc2 directly rather than y_pred which has already
        had the softmax applied.
        '''
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=layer_fc2, labels=y_true)

        '''
        We have now calculated the cross-entropy for each of the image
        classifications so we have a measure of how well the model performs on
        each image individually. But in order to use the cross-entropy to guide
        the optimization of the model's variables we need a single scalar
        value, so we simply take the average of the cross-entropy for all the
        image classifications.'''
        cost = tf.reduce_mean(cross_entropy)

        # Optimization Method

        '''
        Now that we have a cost measure that must be minimized, we can then
        create an optimizer. In this case it is the AdamOptimizer which is an

        advanced form of Gradient Descent.

        Note that optimization is not performed at this point. In fact, nothing
        is calculated at all, we just add the optimizer-object to the
        TensorFlow graph for later execution.
        '''
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

        # Performance Measures

        '''
        We need a few more performance measures to display the progress to the
        user.

        This is a vector of booleans whether the predicted class equals the
        true class of each image.
        '''
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)

        '''
        This calculates the classification accuracy by first type-casting the
        vector of booleans to floats, so that False becomes 0 and True becomes
        1, and then calculating the average of these numbers.
        '''
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # xxx

        
def new_weights(shape):
    """Create new TensorFlow weight variable.

    Function for creating new TensorFlow variables in the given shape and
    initializing them with random values. Note that the initialization is not
    actually done at this point, it is merely being defined in the
    TensorFlow graph.

    Args:
        shape: Shape

    Returns:
        None

    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    """Create new TensorFlow bias variable.

    Function for creating new TensorFlow variables in the given shape and
    initializing them with random values. Note that the initialization is not
    actually done at this point, it is merely being defined in the
    TensorFlow graph.

    Args:
        shape: Shape

    Returns:
        None

    """
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(
        previous_layer, num_input_channels, filter_size, num_filters,
        use_pooling=True):
    """Create new convolutional layer.

    Args:
        previous_layer: The previous layer.
        num_input_channels: Num. channels in prev. layer.
        filter_size: Width and height of each filter.
        num_filters: Number of filters.
        use_pooling: Use 2x2 max-pooling if True.

    Returns:
        (layer, weights): The resulting layer and the filter-weights

    """
    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=previous_layer,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def flatten_layer(layer):
    """Flatten convolutional layer.

    Helper-function for flattening a layer:

    A convolutional layer produces an output tensor with 4 dimensions. We will
    add fully-connected layers after the convolution layers, so we need to
    reduce the 4-dim tensor to 2-dim which can be used as input to the
    fully-connected layer.

    Args:
        layer: Layer to be flattened

    Returns:
        (layer_flat, num_features): The flattened layer and the number of
            features.

    """
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(previous_layer, num_inputs, num_outputs, use_relu=True):
    """Flatten convolutional layer.

    This function creates a new fully-connected layer in the computational
    graph for TensorFlow. Nothing is actually calculated here, we are just
    adding the mathematical formulas to the TensorFlow graph.

    It is assumed that the input is a 2-dim tensor of shape
    [num_images, num_inputs]. The output is a 2-dim tensor of shape
    [num_images, num_outputs].

    Args:
        previous_layer: The previous layer.
        num_inputs: Number of inputs from previous layer.
        num_outputs: Number of outputs.
        use_relu: Use Rectified Linear Unit (ReLU) if True

    Returns:
        layer: New layer

    """
    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(previous_layer, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


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

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(image_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = 'True: {0}'.format(cls_true[i])
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def main():
    """Run main function."""
    # Initialize key variables
    cnn = ConvolutionalNeuralNetwork()


if __name__ == "__main__":
    main()
