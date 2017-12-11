#!/usr/bin/env python3
"""Script to demonstrate  basic tensorflow machine learning."""

# Standard imports
import os
import sys

# PIP3 imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def conv2d(x, W):
    """Create the neural network model.

    Args:
        data: Data

    Returns:
        output: Output

    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    """Create the neural network model.

    Args:
        data: Data

    Returns:
        output: Output

    """
    #                        size of window         movement of window
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(_data, n_classes, keep_rate):
    """Create the neural network model.

    Args:
        _data: Data

    Returns:
        output: Output

    """
    # Initialize key variables
    nodes_conv1 = 32
    nodes_conv2 = 64
    nodes_fc = 1024
    factor = 7 * 7

    weights = {'W_conv1': tf.Variable(tf.random_normal(
                   [5, 5, 1, nodes_conv1])),
               'W_conv2': tf.Variable(tf.random_normal(
                   [5, 5, nodes_conv1, nodes_conv2])),
               'W_fc': tf.Variable(tf.random_normal(
                   [factor * nodes_conv2, nodes_fc])),
               'out': tf.Variable(tf.random_normal([nodes_fc, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([nodes_conv1])),
              'b_conv2': tf.Variable(tf.random_normal([nodes_conv2])),
              'b_fc': tf.Variable(tf.random_normal([nodes_fc])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    data = tf.reshape(_data, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(data, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, factor * nodes_conv2])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


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

    # Set the number of feature sets (numbers) at a time to feed into
    # the neural network
    batch_size = 100

    # Setup placeholder values. Define the expected shapes of input data
    # x is the mnist image
    # y is the label of the image
    x = tf.placeholder('float', [None, width * height])
    y = tf.placeholder('float')

    # Create directory
    directory = '/tmp/nmist'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Populate the directory with data
    mnist = input_data.read_data_sets(directory, one_hot=True)

    # Opimize the cost of the prediction
    prediction = convolutional_neural_network(x, n_classes, keep_rate)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Run the learning
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Train the data
        for epoch in range(epochs_to_try):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', epochs_to_try, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval(
            {x: mnist.test.images[0:desired_test_sample_size],
             y: mnist.test.labels[0:desired_test_sample_size]}))


if __name__ == "__main__":
    main()
