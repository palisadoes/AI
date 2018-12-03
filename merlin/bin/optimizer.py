#!/usr/bin/env python3
"""Script to forecast data using RNN AI using GRU feedback."""

# Standard imports
from __future__ import print_function
import argparse
import time

# PIP3 imports
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Merlin imports
from merlin.model import RNNGRU


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
        '-p', '--periods', help='Lookahead periods.',
        type=int, required=True)
    args = parser.parse_args()
    filename = args.filename
    lookahead_periods = [args.periods]

    '''
    We will use a sequence-length of 1344, which means that each random
    sequence contains observations for 8 weeks. One time-step corresponds to
    one hour, so 24 x 7 time-steps corresponds to a week, and 24 x 7 x 8
    corresponds to 8 weeks.
    '''
    weeks = 5
    sequence_lengths = [weeks * 4]

    # Initialize parameters
    space = {
        'units': hp.choice('units', [512, 256]),
        'dropout': hp.choice('dropout', [0.5]),
        'layers': hp.choice('layers', [1]),
        'sequence_length': hp.choice('sequence_length', sequence_lengths),
        'patience': hp.choice('patience', [1]),
        'batch_size': hp.choice('batch_size', [250]),
        'epochs': hp.choice('epochs', [20])
    }

    # Do training
    rnn = RNNGRU(filename, lookahead_periods)

    # Run trials
    trials = Trials()
    best = fmin(
        rnn.objective, space, algo=tpe.suggest, trials=trials, max_evals=100)

    # Print results
    print(best)
    print(trials.best_trial)

    # Cleanup
    rnn.cleanup()

    '''
    Calculate the duration
    '''

    duration = int(time.time()) - ts_start

    # Print duration
    print("> Training Duration: {}s".format(duration))


if __name__ == "__main__":
    main()
