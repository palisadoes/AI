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
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

class Data(object):
    """Process data for ingestion."""

    def __init__(self, data, features=30, periods=288, forecast=1, base=5):
        """Method that instantiates the class.

        Args:
            data: Dict of values keyed by timestamp
            periods: Number of timestamp data points per vector
            features: Number of features per vector
            forecast: Forecast horizon
            base: Round up to the base int(X)

        Returns:
            None

        """
        # Initialize key variables
        self._features = features
        self._forecast = forecast
        self._base = base
        self._values = []
        values = []

        # Create easier to use lists of data to work with
        for _, value in sorted(data.items()):
            values.append(value)

        # Find the maxes in value every X periods
        item_count = len(values)
        for index in range(0, item_count - periods + 1):
            sample = values[index:index + periods]
            max_sample = max(sample)
            self._values.append(max_sample)

    def load(self):
        """Load RNN data.

            Based on tutorial at:
            http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        sequence_length = self._features + 1
        result = []

        # Create an numpy array of values rounded to the nearest
        # self._base value
        for index in range(len(self._values) - sequence_length):
            result.append(self._values[index: index + sequence_length])
        result = np.array(result)
        for _ in np.nditer(result, op_flags=['readwrite']):
            _[...] = roundup(_, self._base)

        # Use the first 90% of result values as training data
        row = round(0.9 * result.shape[0])
        train = result[:int(row), :]
        np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]

        # Use the last 10% of result values as test data
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1]

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Return
        return (x_train, y_train, x_test, y_test)


def read_file(filename, ts_start=None, rrd_step=300):
    """Read data from file.

    Args:
        filename: Name of CSV file to read
        ts_start: Starting timestamp for which data should be retrieved
        rrd_step: Default RRD step time of the CSV file

    Returns:
        data_dict: Dict of values keyed by timestamp

    """
    # Initialize key variables
    data_dict = {}
    now = _normalize(int(time.time()), rrd_step)
    count = 1

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

    # Back track from the end of the data and delete any zero values until
    # no zeros are found. Sometimes the most recent csv data is zero due to
    # update delays. Replace the deleted entries with a zero value at the
    # beginning of the series
    _timestamps = sorted(data_dict.keys(), reverse=True)
    for timestamp in _timestamps:
        if bool(data_dict[timestamp]) is False:
            data_dict.pop(timestamp, None)
            # Replace the popped item with one at the beginning of the series
            data_dict[int(ts_min - (count * rrd_step))] = 0
        else:
            break
        count += 1

    # Return
    print('Records ingested:', len(data_dict))
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


def roundup(value, base):
    """Round value to nearest base value.

    Args:
        value: Value to round up
        base: Base to use

    Returns:
        result

    """
    # Initialize key variables
    _base = int(base)

    # Return
    _result = int(base * round(float(value)/base))
    result = np.float32(_result)
    return result


def build_model(layers):
    """Build an LSTM model.

    Args:
        layers: List of layer parameters

    Returns:
        model: Graph of LSTM model

    """
    # Initialize key variables
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    print('> Compilation Time:', time.time() - start)
    return model


def main():
    """Main Function.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    periods = 288           # No. of periods per vector for predictions
    features = 30           # No. of features per vector
    forecast = 1            # No. of periods into the future for predictions
    base = 5                # Round up to the base int(X)
    epochs = 10
    batch_size = 128
    rrd_step = 300
    forecast = 5

    # Get filename
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='Name of CSV file to read.',
        type=str, required=True)
    args = parser.parse_args()
    filename = args.filename

    # Open file and get data
    print('> Loading data... ')
    csv_data = read_file(filename, rrd_step=rrd_step)
    data = Data(csv_data, forecast=forecast,
                base=base, periods=periods, features=features)
    (x_train, y_train, x_test, y_test) = data.load()

    print('> Data Loaded. Compiling...', x_train.shape, y_train.shape)
    model = build_model([1, features, 512, 1])
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.05)

    print('->', x_train.shape, x_test.shape)
    #score, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    predicted = model.predict(x_test)
    predicted = np.reshape(predicted, (predicted.size,))

    matching = 0
    for index, prediction in enumerate(predicted):
        value = roundup(prediction, base)
        expected = roundup(y_test[index], base)
        print(value, expected)
        if value == expected:
            matching += 1

    '''matching = 0
    for index, vector in enumerate(x_test):
        _value = model.predict(vector[np.newaxis, :, :])[0, 0]
        value = roundup(_value, base)
        expected = roundup(y_test[index], base)
        print(index, len(x_test), expected, value, _value)
        #print(type(expected), type(value))
        #sys.exit(0)
        if value == expected:
            matching += 1'''

    print('Accuracy: {}'.format(100 * matching/len(x_test)))
    #print('Score: ', score)


if __name__ == "__main__":
    main()
