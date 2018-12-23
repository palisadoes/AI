#!/usr/bin/env python3
"""Script to forecast data using RNN AI using GRU feedback."""

# Standard imports
from __future__ import print_function
import argparse
import time
import sys


# Merlin imports
from merlin.model import RNNGRU
from merlin.database import DataGRU, DataFile


def main():
    """Generate forecasts.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    periods_per_day = 1
    min_sequence_length = 50
    ts_start = int(time.time())

    # Set logging level - No Tensor flow messages
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Get CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='Name of CSV file to read.',
        type=str, required=True)
    parser.add_argument(
        '-b', '--batch-size', help='Size of batch. Default 500.',
        type=int, default=500)
    parser.add_argument(
        '-d', '--days',
        help='Number of days of data to consider when learning. Default 365.',
        type=float, default=365)
    parser.add_argument(
        '-e', '--epochs',
        help='Number of epoch iterations to use. Default 20.',
        type=int, default=20)
    parser.add_argument(
        '-p', '--patience',
        help=(
            'Number of iterations of without loss improvement '
            'before exiting.'),
        type=int, default=10)
    parser.add_argument(
        '-l', '--layers',
        help='Number of GRU layers to use.',
        type=int, default=2)
    parser.add_argument(
        '-o', '--dropout',
        help='Dropout rate as decimal from 0 to 1. Default 0.5 (or 50%)',
        type=float, default=0.5)
    parser.add_argument(
        '-t', '--test_size',
        help=(
            'Test size as decimal fraction of total dataset. '
            'Default 0.2 (or 20%)'),
        type=float, default=0.2)
    parser.add_argument(
        '-u', '--units',
        help='Number of units per layer. Default 512',
        type=int, default=512)
    parser.add_argument(
        '--display',
        help='Display on screen if True. Default False.',
        action='store_true')
    parser.add_argument(
        '--binary',
        help='Predict up/down versus actual values if True. Default False.',
        action='store_true')
    args = parser.parse_args()
    binary = args.binary
    filename = args.filename
    display = args.display
    dropout = args.dropout
    layers = max(0, args.layers)
    patience = args.patience
    test_size = args.test_size
    units = args.units

    '''
    We will use a large batch-size so as to keep the GPU near 100% work-load.
    You may have to adjust this number depending on your GPU, its RAM and your
    choice of sequence_length below.

    Batch size is the number of samples per gradient update. It is a set of N
    samples. The samples in a batch are processed independently, in parallel.
    If training, a batch results in only one update to the model.

    In other words, "batch size" is the number of samples you put into for each
    training round to calculate the gradient. A training round would be a
    "step" within an epoch.

    A batch generally approximates the distribution of the input data better
    than a single input. The larger the batch, the better the approximation;
    however, it is also true that the batch will take longer to process and
    will still result in only one update. For inference (evaluate/predict),
    it is recommended to pick a batch size that is as large as you can afford
    without going out of memory (since larger batches will usually result in
    faster evaluating/prediction).
    '''

    batch_size = args.batch_size

    '''
    We will use a sequence-length of 1344, which means that each random
    sequence contains observations for 8 weeks. One time-step corresponds to
    one hour, so 24 x 7 time-steps corresponds to a week, and 24 x 7 x 8
    corresponds to 8 weeks.
    '''
    days = args.days
    _sequence_length = int(periods_per_day * days)
    sequence_length = max(min_sequence_length, _sequence_length)
    if sequence_length == min_sequence_length:
        print(
            'Days should be > {} for adequate predictions to be made'
            ''.format(int(min_sequence_length/days)))
        sys.exit(0)

    '''
    An epoch is an arbitrary cutoff, generally defined as "one pass over the
    entire dataset", used to separate training into distinct phases, which is
    useful for logging and periodic evaluation.

    Number of epochs to train the model. An epoch is an iteration over the
    entire x and y data provided. Note that in conjunction with initial_epoch,
    epochs is to be understood as "final epoch". The model is not trained for a
    number of iterations given by epochs, but merely until the epoch of index
    epochs is reached.
    '''
    epochs = args.epochs

    # Get the data
    lookahead_periods = [1]

    # Get data from file
    datafile = DataFile(filename)

    # Process data for GRU vector, class creation
    _data = DataGRU(
        datafile, lookahead_periods,
        test_size=test_size, binary=binary)

    if False:
        if binary is True:
            # _data.autocorrelation()
            _data.feature_importance()
            features = _data.suggested_features(count=10, display=True)
            print(features)
            sys.exit(0)

    # Do training
    rnn = RNNGRU(
        _data,
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        layers=layers,
        units=units,
        binary=binary,
        patience=patience,
        display=display)
    rnn.stationary()
    model = rnn.model()
    rnn.save(model)
    rnn.evaluate()

    '''
    Calculate the duration
    '''

    duration = int(time.time()) - ts_start

    '''
    We can now plot an example of predicted output-signals. It is important to
    understand what these plots show, as they are actually a bit more
    complicated than you might think.

    These plots only show the output-signals and not the 20 input-signals used
    to predict the output-signals. The time-shift between the input-signals
    and the output-signals is held fixed in these plots. The model always
    predicts the output-signals e.g. 24 hours into the future (as defined in
    the shift_steps variable above). So the plot's x-axis merely shows how many
    time-steps of the input-signals have been seen by the predictive model so
    far.

    The prediction is not very accurate for the first 30-50 time-steps because
    the model has seen very little input-data at this point. The model
    generates a single time-step of output data for each time-step of the
    input-data, so when the model has only run for a few time-steps, it knows
    very little of the history of the input-signals and cannot make an accurate
    prediction. The model needs to "warm up" by processing perhaps 30-50
    time-steps before its predicted output-signals can be used.

    That is why we ignore this "warmup-period" of 50 time-steps when
    calculating the mean-squared-error in the loss-function. The
    "warmup-period" is shown as a grey box in these plots.

    Let us start with an example from the training-data. This is data that the
    model has seen during training so it should perform reasonably well on
    this data.

    NOTE:

    The model was able to predict the overall oscillations of the temperature
    quite well but the peaks were sometimes inaccurate. For the wind-speed, the
    overall oscillations are predicted reasonably well but the peaks are quite
    inaccurate. For the atmospheric pressure, the overall curve-shape has been
    predicted although there seems to be a slight lag and the predicted curve
    has a lot of noise compared to the smoothness of the original signal.

    '''
    offset = 250
    rnn.plot_train(model, start_idx=rnn.training_rows - offset, length=offset)

    # Example from Test-Set

    '''
    Now consider an example from the test-set. The model has not seen this data
    during training.

    The temperature is predicted reasonably well, although the peaks are
    sometimes inaccurate.

    The wind-speed has not been predicted so well. The daily
    oscillation-frequency seems to match, but the center-level and the peaks
    are quite inaccurate. A guess would be that the wind-speed is difficult to
    predict from the given input data, so the model has merely learnt to output
    sinusoidal oscillations in the daily frequency and approximately at the
    right center-level.

    The atmospheric pressure is predicted reasonably well, except for a lag and
    a more noisy signal than the true time-series.
    '''

    offset = 30
    rnn.plot_test(model, start_idx=rnn.test_rows - offset, length=offset)

    # Cleanup
    rnn.cleanup()

    # Print duration
    print("> Training Duration: {}s".format(duration))


if __name__ == "__main__":
    main()
