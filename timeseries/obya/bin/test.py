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
    percentage_test = 20
    feature_count = 50

    # Import data
    args = arguments()
    df_ = pd.read_csv(args.filename)

    # Preprocessing data
    ary = df_['close'].values

    # Reshape single column array to a list of lists for tensorflow
    if len(ary.shape) == 1:
        ary = ary.reshape(-1, 1)

    # Create test and training data
    rows = ary.shape[0]
    dataset_train = ary[:int(rows * (100 - percentage_test) / 100)]
    dataset_test = ary[
        int(rows * (100 - percentage_test) / 100) - feature_count:]

    # Scale the training data between 0 and 1. Make the test data conform to
    # this range. The test dataset must not influence scaling.
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_train = scaler.fit_transform(dataset_train)
    dataset_test = scaler.transform(dataset_test)

    # Create features for training
    x_train, y_train = features(dataset_train, size=feature_count)
    x_test, y_test = features(dataset_test, size=feature_count)

    # Reshape data for LSTM to have three dimensions namely:
    # (rows, features, columns of data extracted from the dataset)
    # The resulting array makes each row of featues have a row of only a
    # single entry
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_train.shape[0], x_test.shape[1], 1))


def features(ary, size=50):
    """Create feature array using historical data.

    Args:
        ary: Numpy timeseries array
        size: Number of features per row of the array. This is equal to the
            number of historical entries in each row.

    Returns:
        result: feature array

    """
    # Initialize key variables
    historical = []
    actual = []
    rows, _ = ary.shape

    # Process data
    for index in range(size, rows):
        historical.append(ary[index - 50: index, 0])
        actual.append(ary[index, 0])
    historical = np.array(historical)
    actual = np.array(actual)
    result = (historical, actual)
    return result


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
