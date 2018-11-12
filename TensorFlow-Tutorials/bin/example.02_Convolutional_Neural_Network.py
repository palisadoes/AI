#!/usr/bin/env python3
"""MNIST Example Code."""

# Standard imports
import sys
import time
from datetime import timedelta
import math

# PIP imports
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Our custom imports
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

    def __init__(self, train_batch_size=64):
        """Instantiate the class.

        Args:
            train_batch_size: Training batch size

        Returns:
            None

        """
        # Initialize variables
        self.train_batch_size = train_batch_size
        fill = 50

        # Print a spacer
        print('')

        # Counter for total number of iterations performed so far.
        self.total_iterations = 0

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
        self.x_vectors = tf.placeholder(
            tf.float32, shape=[None, self.img_size_flat], name='x')

        '''
        The convolutional layers expect x to be encoded as a 4-dim tensor so we
        have to reshape it so its shape is instead [num_images, img_height,
        img_width, num_channels]. Note that img_height == img_width == img_size
        and num_images can be inferred automatically by using -1 for the size
        of the first dimension. So the reshape operation is:
        '''
        self.x_image = tf.reshape(
            self.x_vectors,
            [-1, self.img_size, self.img_size, self.num_channels])

        print('{0: <{1}} {2}'.format('Encoded X image:', fill, self.x_image))

        '''
        Next we have the placeholder variable for the true labels associated
        with the images that were input in the placeholder variable x. The
        shape of this placeholder variable is [None, num_classes] which means
        it may hold an arbitrary number of labels and each label is a vector of
        length num_classes which is 10 in this case.
        '''
        self.y_true = tf.placeholder(
            tf.float32, shape=[None, self.num_classes], name='y_true')

        '''
        We could also have a placeholder variable for the class-number, but we
        will instead calculate it using argmax. Note that this is a TensorFlow
        operator so nothing is calculated at this point.
        '''
        y_true_cls = tf.argmax(self.y_true, axis=1)

        # Convolutional Layer 1

        '''
        Create the first convolutional layer. It takes x_image as input and
        creates num_filters1 different filters, each having width and height
        equal to filter_size1. Finally we wish to down-sample the image so it
        is half the size by using 2x2 max-pooling.
        '''
        (layer_conv1, self.weights_conv1) = new_conv_layer(
            self.x_image,
            self.num_channels, self.filter_size1, self.num_filters1,
            use_pooling=True)

        '''
        Note: Original image size = 28x28. Convolution reduces this to 14x14.

        Check the shape of the tensor that will be output by the convolutional
        layer. It is (?, 14, 14, 16) which means that there is an arbitrary
        number of images (this is the ?), each image is 14 pixels wide and 14
        pixels high, and there are 16 different channels, one channel for each
        of the filters.
        '''
        print('{0: <{1}} {2}'.format(
            'Number of Channels:', fill, self.num_channels))
        print('{0: <{1}} {2}'.format(
            'First Convolutional Layer:', fill, layer_conv1))

        # Convolutional Layer 2

        '''
        Create the second convolutional layer, which takes as input the output
        from the first convolutional layer. The number of input channels
        corresponds to the number of filters in the first convolutional layer.
        '''
        (layer_conv2, self.weights_conv2) = new_conv_layer(
            layer_conv1,
            self.num_filters1, self.filter_size2, self.num_filters2,
            use_pooling=True)

        '''
        Check the shape of the tensor that will be output by the convolutional
        layer. It is (?, 7, 7, 36) which means that there is an arbitrary
        number of images (this is the ?), each image is 7 pixels wide and 7
        pixels high, and there are 36 different channels, one channel for each
        of the filters.

        Note: Convolution has converted the previously convoluted 14x14 images
        to 7x7 images.
        '''
        print('{0: <{1}} {2}'.format(
            'Second Convolutional Layer:', fill, layer_conv2))

        # Flatten layer

        '''
        The convolutional layers output 4-dim tensors. We now wish to use these
        as input in a fully-connected network, which requires for the tensors
        to be reshaped or flattened to 2-dim tensors.
        '''
        (layer_flat, num_features) = flatten_layer(layer_conv2)

        '''
        Check that the tensors now have shape (?, 1764) which means there's an
        arbitrary number of images which have been flattened to vectors of
        length 1764 each. Note that 1764 = 7 x 7 x 36.
        '''
        print('{0: <{1}} {2}'.format(
            'Flattened Layer:', fill, layer_flat))
        print('{0: <{1}} {2}'.format(
            'Flattened Layer\'s Number of Features: ', fill, num_features))

        # Fully-Connected Layer 1

        '''
        Add a fully-connected layer to the network. The input is the flattened
        layer from the previous convolution. The number of neurons or nodes in
        the fully-connected layer is fc_size. ReLU is used so we can learn
        non-linear relations.
        '''
        layer_fc1 = new_fc_layer(
            layer_flat, num_features, self.fc_size, use_relu=True)

        print('{0: <{1}} {2}'.format(
            'First Fully Connected Layer: ', fill, layer_fc1))

        # Fully-Connected Layer 2

        '''
        Add another fully-connected layer that outputs vectors of length
        10 for determining which of the 10 classes the input image belongs to.
        Note that ReLU is not used in this layer.
        '''
        layer_fc2 = new_fc_layer(
            layer_fc1, self.fc_size, self.num_classes, use_relu=False)

        print('{0: <{1}} {2}'.format(
            'Second Fully Connected Layer: ', fill, layer_fc2))

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
        self.y_pred_cls = tf.argmax(y_pred, axis=1)

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
            logits=layer_fc2, labels=self.y_true)

        '''
        We have now calculated the cross-entropy for each of the image
        classifications so we have a measure of how well the model performs on
        each image individually. But in order to use the cross-entropy to guide
        the optimization of the model's variables we need a single scalar
        value, so we simply take the average of the cross-entropy for all the
        image classifications.
        '''
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
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=1e-4).minimize(cost)

        # Performance Measures

        '''
        We need a few more performance measures to display the progress to the
        user.

        This is a vector of booleans whether the predicted class equals the
        true class of each image.
        '''
        correct_prediction = tf.equal(self.y_pred_cls, y_true_cls)

        '''
        This calculates the classification accuracy by first type-casting the
        vector of booleans to floats, so that False becomes 0 and True becomes
        1, and then calculating the average of these numbers.
        '''
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Create TensorFlow session

        '''
        Once the TensorFlow graph has been created, we have to create a
        TensorFlow session which is used to execute the graph.
        '''
        self.session = tf.Session()

        # Initialize variables

        '''
        The variables for weights and biases must be initialized before we
        start optimizing them.
        '''
        self.session.run(tf.global_variables_initializer())

        # Print a spacer
        print('')

    def optimize(self, num_iterations):
        """Optimize the graph.

        Args:
            num_iterations: Number of optimzation iterations to use.

        Returns:
            None

        """
        # Start-time used for printing time-usage below.
        start_time = time.time()

        for count in range(
                self.total_iterations, self.total_iterations + num_iterations):

            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch, _ = self.data.random_batch(
                batch_size=self.train_batch_size)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {self.x_vectors: x_batch,
                               self.y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            self.session.run(self.optimizer, feed_dict=feed_dict_train)

            # Print status every 100 iterations.
            if count % 100 == 0:
                # Calculate the accuracy on the training-set.
                acc = self.session.run(
                    self.accuracy, feed_dict=feed_dict_train)

                # Print status
                print(
                    'Optimization Iteration: {0:>6}, Training Accuracy: '
                    '{1:>6.1%}'.format(count + 1, acc))

        # Update the total number of iterations performed.
        self.total_iterations += num_iterations

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print('Time usage: ' + str(timedelta(seconds=int(round(time_dif)))))

    def print_test_accuracy(
            self, show_example_errors=False, show_confusion_matrix=False):
        """Print the accuracy of model.

        Args:
            show_example_errors: Show example errors if True.
            show_confusion_matrix: Show confusion matrix if True

        Returns:
            None

        """
        # Split the test-set into smaller batches of this size.
        test_batch_size = 256

        # Number of images in the test-set.
        num_test = self.data.num_test

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_test, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        loop_count = 0

        while loop_count < num_test:
            # The ending index for the next batch is denoted j.
            next_batch_ending_index = min(
                loop_count + test_batch_size, num_test)

            # Get the images from the test-set between index i and j.
            images = self.data.x_test[loop_count:next_batch_ending_index, :]

            # Get the associated labels.
            labels = self.data.y_test[loop_count:next_batch_ending_index, :]

            # Create a feed-dict with these images and labels.
            feed_dict = {self.x_vectors: images,
                         self.y_true: labels}

            # Calculate the predicted class using TensorFlow.
            cls_pred[loop_count:next_batch_ending_index] = self.session.run(
                self.y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batchj to the
            # end-index of the current batch.
            loop_count = next_batch_ending_index

        # Convenience variable for the true class-numbers of the test-set.
        cls_true = self.data.y_test_cls

        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        # Calculate the number of correctly classified images.
        # When summing a boolean array, False means 0 and True means 1.
        correct_sum = correct.sum()

        # Classification accuracy is the number of correctly classified
        # images divided by the total number of images in the test-set.
        acc = float(correct_sum) / num_test

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct_sum, num_test))

        # Plot some examples of mis-classifications, if desired.
        if show_example_errors:
            print('Example errors:')
            self.plot_example_errors(cls_pred=cls_pred, correct=correct)

        # Plot the confusion matrix, if desired.
        if show_confusion_matrix:
            print('Confusion Matrix:')
            self.plot_confusion_matrix(cls_pred=cls_pred)

    def plot_example_errors(self, cls_pred, correct):
        """Plot the TensorFlow example errors.

        Args:
            cls_pred: Array of the predicted class-number for all images in the
                test-set.
            correct: Boolean array whether the predicted class is equal to the
                true class for each image in the test-set.

        Returns:
            None

        """
        # Negate the boolean array.
        incorrect = np.logical_not(correct)

        # Get the images from the test-set that have been
        # incorrectly classified.
        images = self.data.x_test[incorrect]

        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = self.data.y_test_cls[incorrect]

        # Plot the first 9 images.
        plot_images(
            images[0:9], self.img_shape,
            cls_true[0:9], cls_pred=cls_pred[0:9])

    def plot_confusion_matrix(self, cls_pred):
        """Plot the TensorFlow confusion matrix.

        Args:
            cls_pred: Array of the predicted class-number for all images in the
                test-set.

        Returns:
            None

        """
        # Get the true classifications for the test-set.
        cls_true = self.data.y_test_cls

        # Get the confusion matrix using sklearn.
        matrix_of_confusion = confusion_matrix(
            y_true=cls_true, y_pred=cls_pred)

        # Print the confusion matrix as text.
        print(matrix_of_confusion)

        # Plot the confusion matrix as an image.
        plt.matshow(matrix_of_confusion)

        # Make various adjustments to the plot.
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, range(self.num_classes))
        plt.yticks(tick_marks, range(self.num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def plot_conv_weights(self, weights, input_channel=0):
        """Plot the TensorFlow convolution weights.

        Args:
            weights: TensorFlow ops for 4-dim variables e.g. weights_conv1 or
                weights_conv2

        Returns:
            None

        """
        # Retrieve the values of the weight-variables from TensorFlow.
        # A feed-dict is not necessary because nothing is calculated.
        weight_variables = self.session.run(weights)

        # Get the lowest and highest values for the weights.
        # This is used to correct the colour intensity across
        # the images so they can be compared with each other.
        w_min = np.min(weight_variables)
        w_max = np.max(weight_variables)

        # Number of filters used in the conv. layer.
        num_filters = weight_variables.shape[3]

        # Number of grids to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_filters))

        # Create figure with a grid of sub-plots.
        _, axes = plt.subplots(num_grids, num_grids)

        # Plot all the filter-weights.
        for i, axes_object in enumerate(axes.flat):
            # Only plot the valid filter-weights.
            if i < num_filters:
                # Get the weights for the i'th filter of the input channel.
                # See new_conv_layer() for details on the format
                # of this 4-dim tensor.
                img = weight_variables[:, :, input_channel, i]

                # Plot image.
                axes_object.imshow(
                    img, vmin=w_min, vmax=w_max,
                    interpolation='nearest', cmap='seismic')

            # Remove ticks from the plot.
            axes_object.set_xticks([])
            axes_object.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def plot_conv_layer(self, layer, image):
        """Plot the TensorFlow convolution layer.

        Args:
            weights: TensorFlow ops for 4-dim variables e.g. weights_conv1 or
                weights_conv2

        Returns:
            None

        """
        # Create a feed-dict containing just one image.
        # Note that we don't need to feed y_true because it is
        # not used in this calculation.
        feed_dict = {self.x_image: [image]}

        # Calculate and retrieve the output values of the layer
        # when inputting that image.
        values = self.session.run(layer, feed_dict=feed_dict)

        # Number of filters used in the conv. layer.
        num_filters = values.shape[3]

        # Number of grids to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_filters))

        # Create figure with a grid of sub-plots.
        _, axes = plt.subplots(num_grids, num_grids)

        # Plot the output images of all the filters.
        for i, axes_object in enumerate(axes.flat):
            # Only plot the images for valid filters.
            if i < num_filters:
                # Get the output image of using the i'th filter.
                # See new_conv_layer() for details on the format
                # of this 4-dim tensor.
                img = values[0, :, :, i]

                # Plot image.
                axes_object.imshow(img, interpolation='nearest', cmap='binary')

            # Remove ticks from the plot.
            axes_object.set_xticks([])
            axes_object.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def plot_image(self, image):
        """Plot image.

        Args:
            image: Image to plot

        Returns:
            None

        """
        # Adjust the image for viewing
        plt.imshow(
            image.reshape(self.img_shape),
            interpolation='nearest', cmap='binary')

        # Plot the image
        plt.show()


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
        previous_input_layer, num_input_channels, filter_size, num_filters,
        use_pooling=True):
    """Create new convolutional layer.

    This function creates a new convolutional layer in the computational graph
    for TensorFlow. Nothing is actually calculated here, we are just adding the
    mathematical formulas to the TensorFlow graph.

    It is assumed that the input (previous_input_layer) is a 4-dim tensor with
    the following dimensions:

    1) Image number.
    2) Y-axis of each image.
    3) X-axis of each image.
    4) Channels of each image.

    Note that the input channels may either be colour-channels, or it may be
    filter-channels if the input is produced from a previous convolutional
    layer.

    The output is another 4-dim tensor with the following dimensions:

    1) Image number, same as input.
    2) Y-axis of each image. If 2x2 pooling is used, then the height and width
        of the input images is divided by 2.
    3) X-axis of each image. Ditto.
    4) Channels produced by the convolutional filters.

    Args:
        previous_input_layer: The previous layer.new_conv_layer
        num_input_channels: Num. channels in prev. layer.
        filter_size: Width and height of each filter.
        num_filters: Number of filters, which will become the number of output
            channels.
        use_pooling: Use 2x2 max-pooling if True.

    Returns:
        (layer, weights): The resulting layer and the filter-weights

    """
    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.new_conv_layer
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
    layer = tf.nn.conv2d(input=previous_input_layer,
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
    return (layer, weights)


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
    return (layer_flat, num_features)


def new_fc_layer(
        input_vectors, features_per_vector, num_output_features, use_relu=True):
    """Flatten convolutional layer.

    This function creates a new fully-connected layer in the computational
    graph for TensorFlow. Nothing is actually calculated here, we are just
    adding the mathematical formulas to the TensorFlow graph.

    It is assumed that the input is a 2-dim tensor of shape
    [num_images, features_per_vector]. The output is a 2-dim tensor of shape
    [num_images, num_output_features].

    Args:
        input_vectors: The previous layer.
        features_per_vector: Number of faetures per input layer vector.
        num_output_features: Number of outputs.
        use_relu: Use Rectified Linear Unit (ReLU) if True

    Returns:
        layer: New layer

    """
    # Create new weights and biases.
    weights = new_weights(shape=[features_per_vector, num_output_features])
    biases = new_biases(length=num_output_features)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input_vectors, weights) + biases

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
    cnn = ConvolutionalNeuralNetwork()

    # Get the first images from the test-set.
    images = cnn.data.x_test[0:9]

    # Get the true classes for those images.
    cls_true = cnn.data.y_test_cls[0:9]

    # Plot the images and labels using our helper-function above.
    plot_images(images, cnn.img_shape, cls_true=cls_true)

    # Print the accuracy
    cnn.print_test_accuracy()

    # Print accuracy after a number of iterations
    for iterations in [1, 99, 900, 9000]:
        cnn.optimize(iterations)
        cnn.print_test_accuracy()

    # Print test accuracy again
    cnn.print_test_accuracy(
        show_example_errors=True,
        show_confusion_matrix=True)


if __name__ == "__main__":
    main()
