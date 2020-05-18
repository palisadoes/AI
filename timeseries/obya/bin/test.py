#!/usr/bin/env python3
"""Script to forecast timeseries data."""

# Standard imports
from __future__ import print_function
import os
import argparse

# Non-standard imports from ubuntu packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Non-standard imports using pip
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf


def main():
    """Process data.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    history = 50
    epochs = 100

    # Setup parameters
    setup()

    # Import data
    args = arguments()
    df_ = pd.read_csv(args.filename)

    # Preprocessing data
    ary = df_['open'].values

    # Work with model
    model = Model(ary, history=history, epochs=epochs)
    model.train()
    predictions = model.predictions()

    # Plot in matplotlib (All)
    _, ax_ = plt.subplots(figsize=(8, 4))
    plt.plot(ary, color='red', label='Original Stock Price')
    end = len(model.y_train) + history
    ax_.plot(range(end, end + len(predictions)),
             predictions,
             color='blue', label='Predicted Stock Price')
    plt.plot(ary, color='red')
    plt.legend()
    plt.show()

    # Plot in matplotlib (All)
    y_test_scaled = model.scaler.inverse_transform(model.y_test.reshape(-1, 1))
    _, ax_ = plt.subplots(figsize=(8, 4))
    ax_.plot(y_test_scaled, color='red', label='True Price of testing set')
    plt.plot(predictions, color='blue', label='predicted')
    plt.legend()
    plt.show()


class Model():
    """Preprocess the data for models."""

    def __init__(
            self, ary, batch_size=32, history=50, units=96, epochs=50,
            dropout_percent=20, percentage_test=20,
            model_file='/tmp/models.h5'):
        """Create feature array using historical data.

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        self._dropout_rate = dropout_percent / 100
        self._ary = ary
        self._batch_size = batch_size
        self._history = history
        self._units = units
        self._epochs = epochs
        self._percentage_test = percentage_test
        self._model_file = model_file
        self._model = None

        # Preprocess the data
        (self.x_train, self.y_train,
         self.x_test, self.y_test,
         self.scaler) = self._data()

    def create(self):
        """Create Model.

        Args:
            None

        Returns:
            model: LSTM model

        """
        # Create model
        model = Sequential()
        model.add(LSTM(
            units=self._units,
            return_sequences=True,
            input_shape=(self._history, 1)))
        model.add(Dropout(self._dropout_rate))
        model.add(LSTM(units=self._units, return_sequences=True))
        model.add(Dropout(self._dropout_rate))
        model.add(LSTM(units=self._units))
        model.add(Dropout(self._dropout_rate))
        model.add(Dense(1))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def train(self):
        """Train Model.

        Args:
            None

        Returns:
            model: Fitted model

        """
        # Train and save the model
        if os.path.exists(self._model_file) is False:
            model = self.create()
            model.fit(
                self.x_train,
                self.y_train,
                epochs=self._epochs,
                batch_size=self._batch_size)
            model.save(self._model_file)
        else:
            model = self.load()

        # Return
        return model

    def load(self):
        """Load model.

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        model = None

        # Load the model
        if os.path.exists(self._model_file) is True:
            model = load_model(self._model_file)
        return model

    def predictions(self):
        """Load model.

        Args:
            None

        Returns:
            predictions: Predictions

        """
        # Initialize key variables
        predictions = None

        # Get predictions
        model = self.load()
        if bool(model) is True:
            predictions = self.scaler.inverse_transform(
                model.predict(self.x_test))
        return predictions

    def _data(self):
        """Preprocess the data.

        Args:
            None

        Returns:
            None

        """
        # Create the test and training datasets formatted for keras
        process = PreProcessing(
            self._ary,
            percentage_test=self._percentage_test,
            history=self._history)
        result = process.data()
        return result


class PreProcessing():
    """Preprocess the data for models."""

    def __init__(self, ary, percentage_test=20, history=50):
        """Create feature array using historical data.

        Args:
            ary: Numpy timeseries array
            percentage_test: Percentage of the array to use for testing
            history: Number of features per row of the array. This is equal to
                the number of historical entries in each row.


        Returns:
            None

        """
        # Reshape single column array to a list of lists for tensorflow
        if len(ary.shape) == 1:
            ary = ary.reshape(-1, 1)
        self._ary = ary
        self._percentage_test = percentage_test
        self._history = history

        # Create test and training data
        self.rows_total = self._ary.shape[0]
        self.rows_training = int(
            self.rows_total * (100 - percentage_test) / 100)
        self.rows_test = self.rows_total - self.rows_training

    def data(self):
        """Create feature array using historical data.

        Args:
            None

        Returns:
            result: feature array

        """
        # Create a scaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Create test and training data
        dataset_train = self._ary[:self.rows_training]
        dataset_test = self._ary[self.rows_training - self._history:]

        # Scale the training data between 0 and 1. Make the test data
        # conform this range. The test dataset must not influence scaling.
        dataset_train = scaler.fit_transform(dataset_train)
        dataset_test = scaler.transform(dataset_test)

        # Create features for training
        _x_train, y_train = create_features(
            dataset_train, history=self._history)
        _x_test, y_test = create_features(
            dataset_test, history=self._history)

        # Reshape data for LSTM to have three dimensions namely:
        # (rows, features, columns of data extracted from the dataset)
        # The resulting array makes each row of featues have a row of only a
        # single entry (input feature from database)
        x_train = np.reshape(
            _x_train, (self.rows_training - self._history, self._history, 1))
        x_test = np.reshape(
            _x_test, (self.rows_test, self._history, 1))

        # Return
        result = (x_train, y_train, x_test, y_test, scaler)
        return result


def create_features(ary, history=50):
    """Create feature array using historical data.

    Args:
        ary: Numpy timeseries array
        history: Number of features per row of the array. This is equal to the
            number of historical entries in each row.

    Returns:
        result: feature array

    """
    # Initialize key variables
    historical = []
    current = []
    rows, _ = ary.shape

    # Process data
    for index in range(history, rows):
        historical.append(ary[index - history: index, 0])
        current.append(ary[index, 0])
    historical = np.array(historical)
    current = np.array(current)
    result = (historical, current)
    return result


def setup():
    """Setup TensorFlow 2 operating parameters.

    Args:
        None

    Returns:
        None

    """
    # Initialize key variables
    memory_limit = 1024

    # Reduce error logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Limit Tensorflow v2 Limit GPU Memory usage
    # https://www.tensorflow.org/guide/gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if bool(gpus) is True:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for index, _ in enumerate(gpus):
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[index],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print('Physical GPUs: {}, Logical GPUs: {}'.format(
                len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def arguments():
    """Get the CLI arguments.

    Args:
        None

    Returns:
        args: NamedTuple of argument values

    """
    # Get config_dir value
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename',
        help=('Name of file to process.'),
        required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
