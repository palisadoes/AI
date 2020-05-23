#!/usr/bin/env python3
"""Script to forecast timeseries data."""

# Standard imports
from __future__ import print_function
import argparse
from pprint import pprint

# PIP3 packages
import pandas as pd

# Obya imports
from obya.etl import etl
from obya.model import gru


def main():
    """Process data.

    Display data prediction from tensorflow model

    """
    # Initialize key variables

    # Import data
    args = arguments()
    df_ = pd.read_csv(args.filename, names=['timestamp', 'value'], index_col=0)
    data = etl.Data(df_)

    # result = data.split()
    #
    # print('00', '\n', result.x_train.head())
    # print(10, '\n', result.x_train.tail())
    # print(11, '\n', result.x_test.head())
    # print(12, '\n', result.x_test.tail())
    # print(13, '\n', result.x_validate.head())
    # print(14, '\n', result.x_validate.tail())
    # print(20, '\n', result.y_train.tail())
    # print(21, '\n', result.y_test.head())
    # print(22, '\n', result.y_test.tail())
    # print(23, '\n', result.y_validate.head())
    # print(24, '\n', result.x_validate.values[-10:])
    # print(25, '\n', result.y_validate.values[-10:])
    # print(26, '\n', data.vectors().head())
    # print(27, '\n', data.vectors().tail())
    # print(28, '\n', data.vectors().value.values[-10:])

    _model = gru.Model(data)
    _model.info()
    _model.model()


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
