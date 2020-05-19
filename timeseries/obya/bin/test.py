#!/usr/bin/env python3
"""Script to forecast timeseries data."""

# Standard imports
from __future__ import print_function
import argparse
import sys

# PIP3 packages
import pandas as pd

# Obya imports
from obya.etl import etl


def main():
    """Process data.

    Display data prediction from tensorflow model

    """
    # Initialize key variables

    # Import data
    args = arguments()
    df_ = pd.read_csv(args.filename, names=['timestamp', 'value'], index_col=0)
    data = etl.Data(df_)

    result = data.split()
    print(10, '\n', result.x_train.tail())
    print(11, '\n', result.x_test.head())
    print(12, '\n', result.x_test.tail())
    print(13, '\n', result.x_validate.head())

    print(20, '\n', result.y_train.tail())
    print(21, '\n', result.y_test.head())
    print(22, '\n', result.y_test.tail())
    print(23, '\n', result.y_validate.head())

    # print(2, '\n', result[1].head().to_numpy)
    # print(3, '\n', result[2].tail())
    # print(4, '\n', result[3].head())
    # print(5, '\n', result[3].head().to_numpy())
    # print(6, '\n', result[3].head().to_numpy().reshape(1, len(result[3].head())))

    # vectors = data.vectors()
    # print(pd.DataFrame(vectors['value'].head(), columns=['value']))
    # print(vectors.columns)
    # print(vectors.iloc[:, : len(vectors.columns) - 1].head())


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
