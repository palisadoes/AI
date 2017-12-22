#!/usr/bin/env python3
"""Script to forecast data using CNN AI."""

# Standard imports
import sys
from copy import deepcopy
import argparse
import csv
from pprint import pprint
import time
import math

# Pip imports
import numpy as np
import tensorflow as tf


class Data(object):
    """Process data for ingestion."""

    def __init__(self, data, periods=288, forecast=1, nearest=5):
        """Method that instantiates the class.

        Args:
            data: Dict of values keyed by timestamp
            periods: Number of timestamp data points per vector
            forecast: Forecast horizon
            nearest: Round up to the nearest int(X)

        Returns:
            None

        """
        # Initialize key variables
        self._periods = periods
        self._forecast = forecast
        _timestamps = []
        _values = []

        # max_value = roundup(max(list(data.values())), nearest)
        # classes = nearest * 5 # max_value / nearest

        # Create a numpy array for timestamps and values
        for timestamp, value in sorted(data.items()):
            _timestamps.append(timestamp)
            # Round up to nearest X (helps with forecasting accuracy)
            _values.append(roundup(value, nearest))
        self._timestamps = np.asarray(_timestamps)
        self._values = np.asarray(_values)

    def load(self):
        """Load RNN data.

            Based on tutorial at:
            https://mapr.com/blog/deep-learning-tensorflow/

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        ts_length = len(self._timestamps)

        # Create training data
        x_train = self._values[:ts_length - (ts_length % self._periods)]
        y_train = self._values[
            1:ts_length - (ts_length % self._periods) + self._forecast]

        # Create batches
        x_batches = x_train.reshape(-1, self._periods, 1)
        y_batches = y_train.reshape(-1, self._periods, 1)

        # Create test data
        test_x_setup = self._values[
            -(self._periods + self._forecast):]
        x_test = test_x_setup[
            :self._periods].reshape(-1, self._periods, 1)
        y_test = self._values[
            -(self._periods):].reshape(-1, self._periods, 1)

        # Return
        return(x_train, y_train, x_test, y_test, x_batches, y_batches)


def _ingest(filename, ts_start=None):
    """Read data from file.

    Args:
        filename: Name of CSV file to read
        ts_start: Starting timestamp for which data should be retrieved

    Returns:
        None

    """
    # Initialize key variables
    data_dict = {}
    rrd_step = 300
    now = _normalize(int(time.time()), rrd_step)

    # Set the start time to be 2 years by default
    if (ts_start is None) or (ts_start < 0):
        ts_start = now - (3600 * 24 * 365 * 2)
    else:
        ts_start = _normalize(ts_start, rrd_step)

    # Read data
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            timestamp = _normalize(int(row[0]), rrd_step)
            if ts_start <= timestamp:
                value = float(row[1])
                data_dict[timestamp] = value

    # Fill in the blanks in timestamps
    ts_max = max(data_dict.keys())
    ts_min = min(data_dict.keys())
    timestamps = range(
        _normalize(ts_min, rrd_step),
        _normalize(ts_max, rrd_step),
        rrd_step)
    for timestamp in timestamps:
        if timestamp not in data_dict:
            data_dict[timestamp] = 0

    # Return
    return data_dict


def _normalize(timestamp, rrd_step=300):
    """Normalize the timestamp to nearest rrd_step value.

    Args:
        rrd_step: RRD tool step value

    Returns:
        result: Normalized timestamp

    """
    # Return
    result = int((timestamp // rrd_step) * rrd_step)
    return result


def roundup(value, nearest):
    """Round up value to nearest value.

    Args:
        value: Value to round up
        nearest: Nearest integer to round up to

    Returns:
        result

    """
    # Initialize key variables
    _nearest = int(nearest)
    # Return
    _result = int(int(math.ceil(value / _nearest)) * _nearest)
    result = np.float32(_result)
    return result


def main():
    """Main Function.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    periods = 288           # No. of periods per vector for predictions
    forecast = 1            # No. of periods into the future for predictions
    hidden = 100            # No. of neurons to recursively work through.
                            # Can be changed to improve accuracy
    input_vectors = 1       # Number of input vectors submitted
    learning_rate = 0.001   # Small learning rate to not overshoot the minimum
    nearest = 3             # Round up to the nearest int(X)
    epochs_to_try = 1500

    tf.logging.set_verbosity(tf.logging.ERROR)

    # Number of output vectors
    output_vectors = 1

    start = int(time.time())

    # Get filename
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='Name of CSV file to read.',
        type=str, required=True)
    args = parser.parse_args()
    filename = args.filename

    # Open file and get data
    csv_data = _ingest(filename)

    # Get the training and test data
    data_object = Data(
        csv_data, forecast=forecast, nearest=nearest)
    (x_train, y_train,
     x_test, y_test,
     x_batches, y_batches) = data_object.load()

    # ---------------------------------
    # Setup the tensor pathway
    # ---------------------------------
    graph = tf.Graph()
    with graph.as_default():
        # Create variable objects
        x_tensor = tf.placeholder(tf.float32, [None, periods, input_vectors])
        y_tensor = tf.placeholder(tf.float32, [None, periods, output_vectors])

        # Create RNN object
        basic_cell = tf.contrib.rnn.BasicRNNCell(
            num_units=hidden, activation=tf.nn.relu)

        # Choose dynamic over static
        rnn_output, states = tf.nn.dynamic_rnn(
            basic_cell, x_tensor, dtype=tf.float32)

        # Change the form into a tensor
        stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])

        # Specify the type of layer (dense)
        stacked_outputs = tf.layers.dense(stacked_rnn_output, output_vectors)

        # Shape the results
        outputs = tf.reshape(stacked_outputs, [-1, periods, output_vectors])

        # Define the cost function which evaluates the quality of our model
        loss = tf.reduce_sum(tf.square(outputs - y_tensor))

        # Gradient descent method
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Train the result fo the application of the cost function
        training_op = optimizer.minimize(loss)

        '''# Define the metric and update operations
        tf_metric, tf_metric_update = tf.metrics.accuracy(y_tensor,
                                                          p_tensor,
                                                          name="my_metric")

        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")

        # Define initializer to initialize/reset running variables
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)'''

    # ---------------------------------
    # Tensor pathway done
    # ---------------------------------

    # Create the feed_dict
    feed_dict = {x_tensor: x_batches, y_tensor: y_batches}

    # Do learning
    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())

        # Initialize/reset the running variables
        # session.run(running_vars_initializer)

        for epoch in range(epochs_to_try):
            session.run(training_op, feed_dict=feed_dict)
            if epoch % 100 == 0:
                mse = loss.eval(feed_dict=feed_dict)
                print(epoch, '\tMSE:', mse)

        # Round up predictions
        y_predictions = session.run(outputs, feed_dict={x_tensor: x_test})
        y_rounded = deepcopy(y_predictions)
        for x in np.nditer(y_rounded, op_flags=['readwrite']):
            x[...] = roundup(x, nearest)

        # Calculate accuracy
        n_items = y_test.size
        accuracy = (y_test == y_rounded).sum() / n_items
        print('Accuracy:', accuracy)

        # print(y_test)
        # print(y_rounded)


if __name__ == "__main__":
    main()
