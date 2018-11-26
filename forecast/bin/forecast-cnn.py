#!/usr/bin/env python3
"""Script to forecast data using CNN AI."""

# Standard imports
import sys
import argparse
import csv
from pprint import pprint
import time

# Pip imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.models.generator import SequenceGenerator


class Data(object):
    """Process data for ingestion."""

    def __init__(self, data, hours=1, rrd_step=300):
        """Method that instantiates the class.

        Args:
            data: Dict of values keyed by timestamp
            hours:
            rrd_step:

        Returns:
            None

        """
        # Initialize key variables
        interval = hours * 3600
        keys = list(sorted(data.keys()))
        ts_start = min(keys)
        ts_stop = max(keys)
        self._data_dict = {}

        # Get the start and stop boundaries of each bucket
        boundaries = []
        buckets = list(range(ts_start, ts_stop, interval))
        for pointer in range(1, len(buckets)):
            boundaries.append(
                (buckets[pointer - 1], buckets[pointer])
            )

        # Sum the values within the boundaries and set the timestamp to the
        # stop time for the bucket
        for boundary in boundaries:
            start = boundary[0]
            stop = boundary[1]
            values = [0]
            for timestamp in range(start, stop, rrd_step):
                values.append(data[timestamp])
            self._data_dict[stop] = max(values)

    def load(self, width=5, height=6, training_lookahead=1, classes=10):
        """Create histogram of data.

        Args:
            vector_length: Lenth of the vector to use
            training_lookahead: Forecasts should be based on creating
                classifications  X periods into the future
            classes: Number of classes to create

        Returns:
            None

        """
        # Initialize key variables
        vector_length = width * height
        timestamps = list(sorted(self._data_dict.keys()))[:-vector_length]
        max_value = max(list(self._data_dict.values())) * 1.1
        _vectors = []
        _classes = []
        _labels = []
        for index, _ in enumerate(timestamps):
            # Create a list of timestamps to be used by each vector.
            # Make sure each vector meets the minimum length
            vector_timestamps = timestamps[index: index + vector_length]
            if len(vector_timestamps) < vector_length:
                continue

            # Make sure we are not going index beyond the training_lookahead
            if (index + vector_length + training_lookahead) > len(
                    timestamps) - 1:
                continue

            # Create vector
            _vector = []
            for timestamp in vector_timestamps:
                _vector.append(self._data_dict[timestamp])
            _vectors.append(_vector)

            # Create class list for the vector
            pointer = index + vector_length + training_lookahead
            class_timestamp = timestamps[pointer]
            class_value = self._data_dict[class_timestamp]
            _class = [0] * int(classes)
            class_index = int((class_value / max_value) * classes)
            _class[class_index] = 1
            _classes.append(_class)

        np_vectors = np.asarray(_vectors)
        normalized_vectors = np_vectors / np.linalg.norm(np_vectors)

        # Create training and test data
        x_train, x_test, y_train, y_test = train_test_split(
            normalized_vectors, _classes, test_size=0.1, random_state=42)
        return (
            np.asarray(x_train), np.asarray(y_train),
            np.asarray(x_test), np.asarray(y_test))


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


def convolutional_neural_network(width=5, height=6):
    """Create the neural network model.

    Args:
        width: Width of the pseudo image
        height: Height of the pseudo image

    Returns:
        convnet: Output

    """
    # Initialize key variables
    conv1_filter_count = 32
    conv2_filter_count = 64
    fc_units = 1024
    image_height = height
    image_width = width
    filter_size = 2
    pooling_kernel_size = 2
    keep_probability = 0.6
    fully_connected_units = 10

    # Create the convolutional network stuff
    convnet = input_data(
        shape=[None, image_width, image_height, 1], name='input')

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


def _dnn(data, width, height, epochs_to_try):
    """Create Deep Neural Network Model.

    Args:
        data: Data
        width: Width of the pseudo image
        height: Height of the pseudo image
        epochs_to_try: Number of iterations to attempt

    Returns:
        None

    """
    # Initialize key variables
    (_vectors, classes, _test_x, test_y) = data

    # Reshape the test and live vectors
    vectors = _vectors.reshape([-1, width, height, 1])
    test_x = _test_x.reshape([-1, width, height, 1])

    model = tflearn.DNN(
        convolutional_neural_network(width=width, height=height))
    model.fit(
        {'input': vectors},
        {'targets': classes},
        n_epoch=epochs_to_try,
        validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=500,
        show_metric=True,
        run_id='mnist')

    '''print(
        np.round(model.predict([test_x[2]])),
        model.predict([test_x[2]])
    )
    print(test_y[2])'''

    # Evaluate model
    score = model.evaluate(test_x, test_y)
    print('DNN Test accuracy: {:8.2f}%'.format(score[0] * 100))


def main():
    """Main Function.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    epochs_to_try = 3
    start = int(time.time())

    # Define the image width and height of images
    width = 4
    height = 6

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
    data_object = Data(csv_data)
    vectors, classes, test_x, test_y = data_object.load(
        width=width, height=height)

    # Create a data variable
    data = (vectors, classes, test_x, test_y)

    # Get deep neural network results
    _dnn(data, width, height, epochs_to_try)

    # Print duration
    print('Duration:', int(time.time() - start))


if __name__ == "__main__":
    main()
