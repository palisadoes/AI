"""Library to process the ingest of data files."""

# Standard imports
import time
import csv
import os
import sys

# PIP imports
import pandas as pd
import numpy as np


class Dates(object):
    """Convert Pandas date series to components."""

    def __init__(self, dates, date_format):
        """Initialize the class.

        Args:
            dates: Pandas series of dates
            date_format: Date format

        Returns:
            None

        """
        # Get date values from data
        self.weekday = pd.to_datetime(dates, format=date_format).dt.weekday
        self.day = pd.to_datetime(dates, format=date_format).dt.day
        self.dayofyear = pd.to_datetime(
            dates, format=date_format).dt.dayofyear
        self.quarter = pd.to_datetime(dates, format=date_format).dt.quarter
        self.month = pd.to_datetime(dates, format=date_format).dt.month
        self.week = pd.to_datetime(dates, format=date_format).dt.week
        self.year = pd.to_datetime(dates, format=date_format).dt.year


def save_trials(trials, input_filename):
    """Save trial results to file.

    Args:
        trials: List of trial results dicts

    Returns:
        None

    """
    # Initialize key variables
    filename = (
        '/tmp/trials-{}-{}.csv'.format(
            os.path.basename(input_filename), int(time.time())))
    count = 0

    # Iterate
    for trial in trials:
        # Identify variables to output
        hyperparameters = trial['result']['hyperparameters']
        hyperparameters['loss'] = trial['result']['loss']
        hyperparameters['index'] = int(trial['misc']['tid'])

        with open(filename, 'a', newline='') as csvfile:
            fieldnames = list(sorted(hyperparameters.keys()))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Create the header
            if count == 0:
                writer.writeheader()
                count += 1

            # Write data
            writer.writerow(hyperparameters)

    # State what we are doing
    print('\n> Saving results to {}'.format(filename))


def train_validation_test_split(vectors, classes, test_size):
    """Create contiguous (not random) training, validation and test data.

    train_test_split in sklearn.model_selection does this randomly and is
    not suited for time-series data. It also doesn't create a validation-set

    At some point we may want to try Walk Forward Validation methods:

    https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/

    Args:
        vectors: Numpy array of vectors
        classes: Numpy array of classes
        test_size: Percentage of data that is reserved for testing

    Returns:
        result: Training or test vector numpy arrays

    """
    # Make sure we have the correct type of vectors and classes
    '''if isinstance(vectors, np.ndarray) is False or isinstance(
            classes, np.ndarray) is False:
        print('Arguments must of type numpy.ndarray (1)')
        sys.exit(0)'''

    # Initialize key variables
    (rows_train, rows_validation) = _split(vectors.shape[0], test_size)

    # Split training
    x_train = vectors[:rows_train]
    y_train = classes[:rows_train]

    # Split validation
    x_validation = vectors[rows_train:rows_train + rows_validation]
    y_validation = classes[rows_train:rows_train + rows_validation]

    # Split test
    (x_test, y_test) = test_vectors_classes(vectors, classes, test_size)

    # Return
    result = (x_train, x_validation, x_test, y_train, y_validation, y_test)
    return result


def test_vectors_classes(vectors, classes, test_size):
    """Create contiguous (not random) test data.

    train_test_split in sklearn.model_selection does this randomly and is
    not suited for time-series data. It also doesn't create a validation-set

    At some point we may want to try Walk Forward Validation methods:

    https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/

    Args:
        vectors: Dataset of interest
        classes: Classes of interest
        test_size: Percentage of data that is reserved for testing

    Returns:
        x_test: Test vectors

    """
    # Initialize key variables
    (rows_train, rows_validation) = _split(vectors.shape[0], test_size)

    # Make sure we have the correct type of vectors and classes
    '''if isinstance(vectors, np.ndarray) is False or isinstance(
            classes, np.ndarray) is False:
        print('Arguments must of type numpy.ndarray (2)')
        sys.exit(0)'''

    # Split vectors
    x_test = vectors[rows_train + rows_validation:]
    y_test = classes[rows_train + rows_validation:]

    # Return
    return (x_test, y_test)


def _split(rows, test_size):
    """Create dataset allocations for training, validation and test vectors.

    Args:
        rows: Number of rows of data
        test_size: Percentage of data that is reserved for testing

    Returns:
        result: Tuple of training, validation and test row values

    """
    # Initialize key variables
    rows_test = int(test_size * rows)
    rows_validation = int(test_size * rows)
    rows_train = rows - (rows_test + rows_validation)

    # Return
    return (rows_train, rows_validation)


def binary_accuracy(predictions, actuals):
    """Get the accuracy of predictions versus actuals.

    Args:
        predictions: np.array of predictions (floats)
        actuals: np.array of actual values (ints) of either 1 or -1

    Returns:
        result: Float of accuracy

    """
    # Calculate average accuracy
    _predictions = to_buy_sell(predictions.flatten().tolist())
    _actuals = actuals.flatten().astype(int).tolist()
    sameness = (_actuals == _predictions).astype(int).tolist()
    result = sum(sameness)/len(sameness)

    # Print accuracy result lists to aid visualization of the data
    print('> Actuals: {}'.format(_actuals))
    print('> Predicted: {}'.format(_predictions))

    print(
        '> Predicted (Original Formatting): {}'.format(predictions.flatten()))
    p_list = _predictions.astype(int).tolist()
    print('> Average Predicted Value: {:.3f}'.format(sum(p_list)/len(p_list)))

    # Return
    return result


def _binary_accuracy(predictions, limit=0.3):
    """Save trial results to file.

    Args:
        trials: List of trial results dicts

    Returns:
        result: Numpy arrary of results

    """
    higher = (predictions > 1 - limit).astype(int) * 1
    lower = (predictions < limit).astype(int) * 0

    above_lower_bound = (predictions > limit).astype(int) * -1
    below_upper_bound = (predictions < 1 - limit).astype(int) * -1
    undetermined = above_lower_bound * below_upper_bound

    result = higher + lower + undetermined
    print(higher, '\n')
    print(lower, '\n')
    print(undetermined)
    items = []
    for item in predictions:
        items.append(item)
    print(items)
    sys.exit(0)
    return result


def to_buy_sell(values):
    """Convert list of floats to list of +1 or -1 values depending on sign.

    Args:
        values: List of list of floats

    Returns:
        result: Numpy arrary of results

    """
    # Initialize key variables
    buy = (np.array(values) > 0).astype(int) * 1
    sell = (np.array(values) <= 0).astype(int) * -1
    result = buy + sell
    return result
