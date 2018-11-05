#!/usr/bin/env python3
"""MNIST Example Code."""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from mnist import MNIST


def plot_images(images, cls_true, cls_pred=None):
    """Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image."""

    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _ = data.random_batch(batch_size=batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)


def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))


def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = data.y_test_cls

    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.x_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.y_test_cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def main():
    """Main Function."""
    # Get data from files
    data = MNIST(data_dir='data/MNIST/')

    # The images are stored in one-dimensional arrays of this length.
    img_size_flat = data.img_size_flat

    # Tuple with height and width of images used to reshape arrays.
    img_shape = data.img_shape

    # Number of classes, one class for each of 10 digits.
    num_classes = data.num_classes

    # Get the first images from the test-set.
    images = data.x_test[0:9]

    # Get the true classes for those images.
    cls_true = data.y_test_cls[0:9]

    # Plot the images and labels using our helper-function above.
    plot_images(images=images, cls_true=cls_true)

    # Setup tensors

    '''
    First we define the placeholder variable for the input images. This
    allows us to change the images that are input to the TensorFlow graph.
    This is a so-called tensor, which just means that it is a
    multi-dimensional vector or matrix. The data-type is set to float32 and
    the shape is set to [None, img_size_flat], where None means that the
    tensor may hold an arbitrary number of images with each image being a
    vector of length img_size_flat.
    '''
    x = tf.placeholder(tf.float32, [None, img_size_flat])

    '''
    Next we have the placeholder variable for the true labels associated
    with the images that were input in the placeholder variable x. The shape
    of this placeholder variable is [None, num_classes] which means it may
    hold an arbitrary number of labels and each label is a vector of length
    num_classes which is 10 in this case.
    '''
    y_true = tf.placeholder(tf.float32, [None, num_classes])

    '''
    Finally we have the placeholder variable for the true class of each
    image in the placeholder variable x. These are integers and the
    dimensionality of this placeholder variable is set to [None] which means
    the placeholder variable is a one-dimensional vector of arbitrary
    length.
    '''
    y_true_cls = tf.placeholder(tf.int64, [None])

    '''
    Apart from the placeholder variables that were defined above and which
    serve as feeding input data into the model, there are also some model
    variables that must be changed by TensorFlow so as to make the model
    perform better on the training data.

    The first variable that must be optimized is called weights and is defined
    here as a TensorFlow variable that must be initialized with zeros and whose
    shape is [img_size_flat, num_classes], so it is a 2-dimensional tensor
    (or matrix) with img_size_flat rows and num_classes columns.
    '''
    weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))

    '''
    The second variable that must be optimized is called biases and is defined
    as a 1-dimensional tensor (or vector) of length num_classes.
    '''
    biases = tf.Variable(tf.zeros([num_classes]))

    # Create the model

    '''
    This simple mathematical model multiplies the images in the placeholder
    variable x with the weights and then adds the biases.

    The result is a matrix of shape [num_images, num_classes] because x has
    shape [num_images, img_size_flat] and weights has shape
    [img_size_flat, num_classes], so the multiplication of those two matrices
    is a matrix with shape [num_images, num_classes] and then the biases vector
    is added to each row of that matrix.
    '''
    logits = tf.matmul(x, weights) + biases

    '''
    Now logits is a matrix with num_images rows and num_classes columns, where
    the element of the $i$'th row and j'th column is an estimate of how
    likely the i'th input image is to be of the j'th class.

    However, these estimates are a bit rough and difficult to interpret because
    the numbers may be very small or large, so we want to normalize them so
    that each row of the logits matrix sums to one, and each element is limited
    between zero and one. This is calculated using the so-called softmax
    function and the result is stored in y_pred.
    '''
    y_pred = tf.nn.softmax(logits)

    '''
    The predicted class can be calculated from the y_pred matrix by taking the
    index of the largest element in each row.
    '''
    y_pred_cls = tf.argmax(y_pred, axis=1)

    # Cost-function to be optimized

    '''
    To make the model better at classifying the input images, we must somehow
    change the variables for weights and biases. To do this we first need to
    know how well the model currently performs by comparing the predicted
    output of the model y_pred to the desired output y_true.

    The cross-entropy is a performance measure used in classification. The
    cross-entropy is a continuous function that is always positive and if the
    predicted output of the model exactly matches the desired output then the
    cross-entropy equals zero. The goal of optimization is therefore to
    minimize the cross-entropy so it gets as close to zero as possible by
    changing the weights and biases of the model.

    TensorFlow has a built-in function for calculating the cross-entropy. Note
    that it uses the values of the logits because it also calculates the
    softmax internally.
    '''
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=y_true)

    '''
    We have now calculated the cross-entropy for each of the image
    classifications so we have a measure of how well the model performs on each
    image individually. But in order to use the cross-entropy to guide the
    optimization of the model's variables we need a single scalar value, so we
    simply take the average of the cross-entropy for all the image
    classifications.
    '''
    cost = tf.reduce_mean(cross_entropy)

    # Optimization method

    '''
    Now that we have a cost measure that must be minimized, we can then create
    an optimizer. In this case it is the basic form of Gradient Descent where
    the step-size is set to 0.5.

    Note that optimization is not performed at this point. In fact, nothing is
    calculated at all, we just add the optimizer-object to the TensorFlow graph
    for later execution.
    '''
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

    # Performance measures

    '''
    We need a few more performance measures to display the progress to the
    user.

    This is a vector of booleans whether the predicted class equals the true
    class of each image.
    '''
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    '''
    This calculates the classification accuracy by first type-casting the
    vector of booleans to floats, so that False becomes 0 and True becomes 1,
    and then calculating the average of these numbers.
    '''
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


if __name__ == "__main__":
    main()
