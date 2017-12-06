#!/usr/bin/env python3
"""Script to demonstrate  basic tensorflow machine learning."""

# Standard imports
import os

# PIP3 imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def neural_network_model(data):
    """Create the neural network model.

    Args:
        data: Data

    Returns:
        output: Output

    """
    # Define the image width and height of images
    width = 28
    height = 28

    # Initialize key variables (hidden layer node counts)
    n_nodes_hl1 = 500
    n_nodes_hl2 = 500
    n_nodes_hl3 = 500

    # Set the number of classes (10 numbers)
    n_classes = 10

    # Create the nodes in the neural network
    hidden_1_layer = {
        'weights': tf.Variable(tf.random_normal([height * width, n_nodes_hl1])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))}

    # Calculate the values at each node
    # (input_data * weights) + biases
    l1_nodes = tf.add(tf.matmul(
        data, hidden_1_layer['weights']), hidden_1_layer['biases'])

    # Convert node values using rectified linear units for activation
    l1_nodes = tf.nn.relu(l1_nodes)

    # Do the same for other layers
    l2_nodes = tf.add(tf.matmul(
        l1_nodes, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2_nodes = tf.nn.relu(l2_nodes)
    l3_nodes = tf.add(tf.matmul(
        l2_nodes, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3_nodes = tf.nn.relu(l3_nodes)

    # Compute the output and return
    output = tf.matmul(
        l3_nodes, output_layer['weights']) + output_layer['biases']
    return output


def main():
    """Main Function.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    epochs_to_try = 10

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
    prediction = neural_network_model(x)
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
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


if __name__ == "__main__":
    main()
