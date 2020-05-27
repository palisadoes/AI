'''Module to prepare memory for safe model creation.'''

import os
from collections import namedtuple

import tensorflow as tf


def _device_name(item):
    """Create a device name for Tensorflow.

    Args:
        item: tensorflow.python.eager.context.PhysicalDevice

    Returns:
        result: Name of device for tensorflow

    """
    # Process
    components = item.split(':')
    result = '/{}'.format(':'.join(components[1:]))
    return result


def setup():
    """Setup TensorFlow 2 operating parameters.

    Args:
        None

    Returns:
        result: Processor namedtuple of GPUs and CPUs in the system

    """
    # Initialize key variables
    memory_limit = 1024
    Processors = namedtuple('Processors', 'gpus, cpus')
    gpu_names = []
    cpu_names = []

    # Reduce error logging
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Limit Tensorflow v2 Limit GPU Memory usage
    # https://www.tensorflow.org/guide/gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')
    if bool(gpus) is True:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for _, gpu in enumerate(gpus):
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit)])
                gpu_names.append(_device_name(gpu.name))

            # Currently, memory growth needs to be the same across GPUs
            for _, cpu in enumerate(cpus):
                cpu_names.append(_device_name(cpu.name))

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Return
    result = Processors(gpus=gpu_names, cpus=cpu_names)
    return result
