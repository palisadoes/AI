'''Module to prepare memory for safe model creation.'''

import os
import tensorflow as tf


def setup():
    """Setup TensorFlow 2 operating parameters.

    Args:
        None

    Returns:
        None

    """
    # Initialize key variables
    memory_limit = 1024

    # Reduce error logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Limit Tensorflow v2 Limit GPU Memory usage
    # https://www.tensorflow.org/guide/gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if bool(gpus) is True:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for index, _ in enumerate(gpus):
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[index],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print('Physical GPUs: {}, Logical GPUs: {}'.format(
                len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
