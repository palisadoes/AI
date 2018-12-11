#!/usr/bin/env python3
"""Script to forecast data using RNN AI using GRU feedback."""

# Standard imports
from __future__ import print_function
import argparse
import time
import sys
from pprint import pprint

# PIP3 imports
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Merlin imports
from merlin.model import RNNGRU
from merlin.database import DataGRU
from merlin.general import save_trials


def main():
    """Generate forecasts.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    ts_start = int(time.time())

    # Set logging level - No Tensor flow messages
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Get CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='Name of CSV file to read.',
        type=str, required=True)
    parser.add_argument(
        '-m', '--max_evals',
        help=('Maximum number of experiments before stopping. Default 500'),
        type=int, default=500)
    parser.add_argument(
        '-p', '--periods', help='Lookahead periods.',
        type=int, required=True)
    parser.add_argument(
        '--binary',
        help='Predict up/down versus actual values if True. Default False.',
        action='store_true')
    parser.add_argument(
        '-t', '--test_size',
        help=(
            'Test size as decimal fraction of total dataset. '
            'Default 0.2 (or 20%)'),
        type=float, default=0.2)
    args = parser.parse_args()
    filename = args.filename
    binary = args.binary
    max_evals = args.max_evals
    test_size = args.test_size
    lookahead_periods = [args.periods]

    '''
    We will use a sequence-length of 1344, which means that each random
    sequence contains observations for 8 weeks. One time-step corresponds to
    one hour, so 24 x 7 time-steps corresponds to a week, and 24 x 7 x 8
    corresponds to 8 weeks.
    '''
    periods_per_day = 1
    days_per_week = 7
    sequence_lengths = [
        periods_per_day * days_per_week * 12,
        periods_per_day * days_per_week * 24]

    # Initialize parameters
    space = {
        'units': hp.choice('units', list(range(500, 505, 5))),
        'dropout': hp.choice('dropout', [0.5]),
        'layers': hp.choice('layers', [1]),
        'sequence_length': hp.choice('sequence_length', sequence_lengths),
        'patience': hp.choice('patience', [5]),
        'batch_size': hp.choice('batch_size', [500]),
        'epochs': hp.choice('epochs', [1000])
    }

    _data = DataGRU(
        filename, lookahead_periods,
        test_size=test_size, binary=binary)

    # Do training
    rnn = RNNGRU(_data)

    # Test for stationarity
    if rnn.stationary() is False:
        print(
            '> Data appears to be a random walk and is not suitable '
            'for forecasting.')
        sys.exit(0)

    # Run trials
    trials = Trials()
    _ = fmin(
        rnn.objective, space,
        algo=tpe.suggest, trials=trials, max_evals=max_evals)

    # Print results
    print('\n> Best Trial:\n')
    pprint(trials.best_trial)

    best_index = trials.best_trial['misc']['tid']
    print('\n> Best Trial {}:\n'.format(best_index))
    pprint(trials.trials[best_index])

    # Cleanup
    rnn.cleanup()

    # Write trial results to file
    save_trials(trials.trials, filename)

    '''
    Calculate the duration
    '''

    duration = int(time.time()) - ts_start

    # Print duration
    print("\n> Training Duration: {}s".format(duration))


if __name__ == "__main__":
    main()
