#!/usr/bin/env python3
"""Script to demonstrate basic tensorflow machine learning."""

# Standard imports
import os
import random
import pickle
from collections import Counter

# PIP imports
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Data(object):
    """Process Sentiment Data."""

    def __init__(self, f_positive, f_negative):
        """Method to instantiate the class.

        Args:
            f_positive: File with positive sentiments
            f_negative: File with negative sentiments

        Returns:
            None

        """
        self.pos = f_positive
        self.neg = f_negative
        self.max_lines = 100000
        self.lemmatizer = WordNetLemmatizer()

        # Create the lexicon
        self.lexicon = self._create_lexicon()

    def _create_lexicon(self):
        """Create the lexicon from files.

        Args:
            None

        Returns:
            result: Output

        """
        lexicon = []
        with open(self.pos, 'r') as f_handle:
            contents = f_handle.readlines()
            for word in contents[:self.max_lines]:
                all_words = word_tokenize(word)
                lexicon += list(all_words)

        with open(self.neg, 'r') as f_handle:
            contents = f_handle.readlines()
            for word in contents[:self.max_lines]:
                all_words = word_tokenize(word)
                lexicon += list(all_words)

        lexicon = [self.lemmatizer.lemmatize(i) for i in lexicon]
        w_counts = Counter(lexicon)
        result = []
        for count in w_counts:
            if 1000 > w_counts[count] > 50:
                result.append(count)
        return result

    def _sample_handling(self, filename, classification):
        """Handle samples from file.

        Args:
            filename: Name of file
            classification: Classification of featureset for lines in file

        Returns:
            result: Output

        """
        featureset = []
        lexicon = self.lexicon

        with open(filename, 'r') as f_handle:
            contents = f_handle.readlines()
            for line_word in contents[:self.max_lines]:
                # Get words on current line
                _current_words = word_tokenize(line_word.lower())
                current_words = [
                    self.lemmatizer.lemmatize(i) for i in _current_words]

                # Create a feature set based on the words found on the line
                features = np.zeros(len(lexicon))
                for word in current_words:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] += 1

                features = list(features)
                featureset.append([features, classification])

        # Return
        return featureset

    def create_feature_sets_and_labels(self, test_size=0.1):
        """Create feature sets and labels.

        Args:
            test_size: Size of test

        Returns:
            result: Output

        """
        _features = []
        _features += self._sample_handling(self.pos, [1, 0])
        _features += self._sample_handling(self.neg, [0, 1])
        random.shuffle(_features)
        features = np.array(_features)

        testing_size = int(test_size * len(features))

        training_vectors = list(features[:, 0][:-testing_size])
        training_labels = list(features[:, 1][:-testing_size])
        test_vectors = list(features[:, 0][-testing_size:])
        test_labels = list(features[:, 1][-testing_size:])

        '''
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)
        Y = one_hot_encode(y)
        a, b, c, d = train_test_split(training_vectors, training_labels, test_size=0.2)

        # Save model
        saver = tf.train.Saver()
        save_file = saver.save(sess, save_directory)
        saver.restore(save_directory)

        '''

        # Return
        return training_vectors, training_labels, test_vectors, test_labels

    def one_hot_encode(self, labels):
        """One hot encode labels.

        Args:
            labels: List of labels

        Returns:
            result: Result

        """
        n_labels = len(labels)
        n_unique_labels = len(np.unique(labels))
        result = np.zeros((n_labels, n_unique_labels))
        result[np.arange(n_labels), labels] = 1
        return result


def neural_network_model(vector, n_classes, vector_length):
    """Create the neural network model.

    Args:
        vector: Vector data
        n_classes: Number of classes
        vector_length: Length of an individual vector

    Returns:
        output: Output

    """
    # Initialize key variables (hidden layer node counts)
    n_nodes_hl1 = 500
    n_nodes_hl2 = 500
    n_nodes_hl3 = 500

    # Create the nodes in the neural network
    hidden_1_layer = {
        'weights': tf.Variable(tf.random_normal([vector_length, n_nodes_hl1])),
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
        vector, hidden_1_layer['weights']), hidden_1_layer['biases'])

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
    n_classes = 2

    # Set the number of feature sets (numbers) at a time to feed into
    # the neural network
    batch_size = 100

    # Read data
    pos = 'data/pos.txt'
    neg = 'data/neg.txt'
    data = Data(pos, neg)
    (training_vectors,
     training_labels,
     test_vectors,
     test_labels) = data.create_feature_sets_and_labels()

    # If you want to save the data
    '''
    with open('/path/to/sentiment_set.pickle','wb') as f:
            pickle.dump([training_vectors,training_labels,test_vectors,test_labels],f)
    '''

    # Get the vector_length
    vector_length = len(training_vectors[0])

    # Setup placeholder values. Define the expected shapes of input data
    # x is the mnist image
    # y is the label of the image
    vector = tf.placeholder(tf.float32, shape=[None, vector_length])
    label = tf.placeholder(tf.float32)

    # Opimize the cost of the prediction
    prediction = neural_network_model(vector, n_classes, vector_length)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=prediction, labels=label))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Run the learning
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Train the data
        for epoch in range(epochs_to_try):
            epoch_loss = 0
            pointer = 0
            while pointer < len(training_vectors):
                start = pointer
                end = pointer + batch_size
                batch_of_vectors = np.array(training_vectors[start:end])
                batch_of_labels = np.array(training_labels[start:end])
                _, batch_loss = sess.run(
                    [optimizer, cost],
                    feed_dict={
                        vector: batch_of_vectors, label: batch_of_labels})
                epoch_loss += batch_loss
                pointer += batch_size

            print(
                'Epoch', epoch, 'completed out of',
                epochs_to_try, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy:', accuracy.eval(
            {vector: test_vectors, label: test_labels}))

        '''
        # YouTube - https://www.youtube.com/watch?v=yX8KuPZCAMo
        # How to predict a value
        new_prediction = sess.run(prediction, feed_dict={vector: vector_value})
        new_accuracy = sess.run(
            accuracy, feed_dict={vector: vector_value, label: label_value})
        '''


if __name__ == '__main__':
    main()
