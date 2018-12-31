"""Library to process the ingest of data files."""

# Standard imports
import time
import csv
import os

# PIP imports
import pandas as pd
from tensorflow.python.client import device_lib
import tensorflow as tf
from keras import backend


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


def train_validation_test_split(vectors, classes, test_size):
    """Create contiguous (not random) training and test data.

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
    # Initialize key variables
    num_data = vectors.shape[0]
    num_test = int(test_size * num_data)
    num_validation = int(test_size * num_data)
    num_train = num_data - (num_test + num_validation)

    # Split vectors
    x_train = vectors[:num_train]
    x_validation = vectors[num_train:num_train + num_validation]
    x_test = vectors[num_train + num_validation:]

    # Split classes
    y_train = classes[:num_train]
    y_validation = classes[num_train:num_train + num_validation]
    y_test = classes[num_train + num_validation:]

    # Return
    result = (x_train, x_validation, x_test, y_train, y_validation, y_test)
    return result


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
