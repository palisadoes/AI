"""Library to process the ingest of data files."""

# Standard imports
import time
import csv
import os
import sys

# PIP imports
import pandas as pd
import numpy as np

from tensorflow.python.client import device_lib
import tensorflow as tf
from keras import backend


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
    _predictions = predictions.flatten()
    _actuals = actuals.flatten()
    sameness = (_actuals == _predictions).astype(int).tolist()
    result = sum(sameness)/len(sameness)

    # Print accuracy result lists to aid visualization of the data
    a_list = _actuals.astype(int).tolist()
    p_list = _predictions.astype(int).tolist()
    print('> Actuals:\n{}'.format(a_list))
    print('> Predicted:\n{}'.format(p_list))
    print('> Average Actual Value: {:.3f}'.format(sum(a_list)/len(a_list)))
    print('> Average Predicted Value: {:.3f}'.format(sum(p_list)/len(p_list)))

    # Return
    return result


def get_available_gpus():
    """Get available number of GPUs.

    Args:
        None

    Returns:
        result: Float of accuracy

    """
    ###################################
    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.1

    # Crash with DeadlineExceeded instead of hanging forever when your
    # queues get full/empty
    config.operation_timeout_in_ms = 60000

    # Create a session with the above options specified.
    backend.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################

    # Create a temporary session to get the data.
    # IMPORTANT - Memory will be immediately freed this way.
    with tf.Session(config=config) as _:
        local_device_protos = device_lib.list_local_devices()

    # Return
    result = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return result


def weights_match(model1, model2):
    """Determine whether the weights of two Keras models match.

    Args:
        model1: First model
        model2: Second model

    Returns:
        result: True if there is a match

    """
    # Get weights
    weights_m1 = backend.batch_get_value(model1.weights)
    weights_m2 = backend.batch_get_value(model2.weights)

    # Evaluate equivalence
    if all([np.all(w == ow) for w, ow in zip(weights_m1, weights_m2)]):
        result = True
    else:
        result = False
    return result


def trim_correlated(df_in, threshold=0.95):
    """Drop Highly Correlated Features.

    Based on: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

    Args:
        df_in: Input dataframe
        threshold: Correlation threshold

    Returns:
        df_out: pd.DataFrame with uncorrelated columns

    """
    # Find index of feature columns with correlation greater than threshold
    to_drop = correlated_columns(df_in, threshold=threshold)
    df_out = df_in.drop(to_drop, axis=1)

    # Return
    return df_out


def correlated_columns(df_in, threshold=0.95):
    """Return correlated feature columns.

    Based on: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

    Args:
        df_in: Input dataframe
        threshold: Correlation threshold

    Returns:
        to_drop: Columns of pd.DataFrame with correlated columns

    """
    # Create correlation matrix
    corr_matrix = df_in.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than threshold
    to_drop = [
        column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop
