#!/usr/bin/env python3
"""Script to forecast timeseries data."""

# Standard imports
from __future__ import print_function
import argparse
import os
import sys

# PIP3 packages
import pandas as pd
import numpy as np

# Obya imports
from obya.etl import etl
from obya.model import gru
from obya.model import plot


def process(df_, f_steps=0, p_steps=0):
    """Pre-process data.

    Args:
        df_: DataFrame to process. Indexed by epoch timestamp

    Returns:
        result: processed DataFrame

    """
    # Look out one week on values
    future_values = df_['value'].rolling(f_steps).max().shift(-f_steps)
    future_values = np.array(future_values)[p_steps:-f_steps]
    print(future_values.shape)

    # Look back one week on time
    # past_maxes = df_.index.to_series().rolling(p_steps).max()
    past_maxes = df_.index.to_series().rolling(p_steps).max()
    past_maxes = np.array(past_maxes).astype('int')[p_steps:-f_steps]
    print(past_maxes.shape)

    # Recreate
    result = pd.DataFrame(future_values, index=past_maxes, columns=['value'])

    # Create a one day moving average
    #result = result.iloc[p_steps:-f_steps].set_index('timestamp')

    # Return
    return result


def main():
    """Process data.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    length = 1000
    steps_per_day = 288
    f_steps = steps_per_day * 7
    p_steps = steps_per_day * 30

    # Import data
    args = arguments()

    # Create an identifier and import data
    identifier = 'test_{}'.format(os.path.basename(args.filename))
    df_ = pd.read_csv(args.filename, names=['timestamp', 'value'], index_col=0)

    print(df_.columns)
    print(df_.head(2))

    # Process data
    df_ = process(df_, f_steps=f_steps, p_steps=p_steps)

    print(df_.columns)
    print(df_)
    # print(df_.tail(2))
    # sys.exit(0)

    # Convert data for training
    data = etl.Data(df_, shift=args.shift)

    # Train if necessary
    if args.train is True:
        # Create Model
        model = gru.Model(
            data,
            identifier,
            batch_size=args.batch_size,
            epochs=args.epochs,
            sequence_length=args.sequence_length,
            dropout=args.dropout,
            divider=1,
            layers=args.layers,
            patience=args.patience,
            units=args.units,
            test_size=args.test_size,
            multigpu=False
        )
        model.info()
        model.mtrain()

    _plot = plot.Plot(data, identifier)
    _plot.history()
    _plot.train(0, length=length)
    _plot.test(0, length=len(data.values()))
    _plot.train_test()


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

    parser.add_argument(
        '-b', '--batch_size', help='Size of batch. Default 500.',
        type=int, default=256)

    parser.add_argument(
        '-e', '--epochs',
        help='Number of epoch iterations to use. Default 20.',
        type=int, default=20)

    parser.add_argument(
        '-p', '--patience',
        help='''\
Number of iterations of without loss improvement before exiting. \
Default 10''',
        type=int, default=5)

    parser.add_argument(
        '-l', '--layers',
        help='Number of GRU layers to use. Default 1',
        type=int, default=1)

    parser.add_argument(
        '-d', '--dropout',
        help='Dropout rate as decimal from 0 to 1. Default 0.1 (10 percent)',
        type=float, default=0.1)

    parser.add_argument(
        '-s', '--test_size',
        help='''\
Test size as decimal fraction of total dataset. Default 0.2 (20 percent)''',
        type=float, default=0.2)

    parser.add_argument(
        '-u', '--units',
        help='Number of units per layer. Default 512',
        type=int, default=128)

    parser.add_argument(
        '-q', '--sequence_length',
        help='Sequence length. Default 10',
        type=int, default=500)

    parser.add_argument(
        '-S', '--shift',
        help='Number of predictions into the future to forecast. Default 1',
        type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
