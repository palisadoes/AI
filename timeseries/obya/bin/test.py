#!/usr/bin/env python3
"""Script to forecast timeseries data."""

# Standard imports
from __future__ import print_function
import argparse

# PIP3 packages
import pandas as pd

# Obya imports
from obya.etl import etl
from obya.model import gru
from obya.model import plot


def main():
    """Process data.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    identifier = 'test'
    length = 1000

    # Import data
    args = arguments()
    df_ = pd.read_csv(args.filename, names=['timestamp', 'value'], index_col=0)
    data = etl.Data(df_)

    # Train if necessary
    if args.train is True:
        # Create Model
        model = gru.Model(
            data, identifier, epochs=10, batch_size=128, units=64)
        model.info()
        model.train()

    _plot = plot.Plot(data, identifier)
    _plot.history()
    _plot.train(0, length=length)
    _plot.test(0, length=length)


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

    parser.add_argument(
        '-t', '--train',
        help=('Train the data model if True.'),
        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
