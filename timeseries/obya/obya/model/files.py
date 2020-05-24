"""File manipulation module."""

from collections import namedtuple

# PIP3 imports
import yaml
from keras.models import model_from_yaml


def files(identifier):
    """Create well known locations for model files.

    Args:
        identifier: Identifier for training set

    Returns:
        _files: NamedTuple of file locations

    """
    # Set key file locations
    path_prefix = '/tmp/obya-{}'.format(identifier)
    Files = namedtuple(
        'Files',
        'checkpoint, model_weights, model_parameters, log_dir, history')
    _files = Files(
        checkpoint='{}.checkpoint.h5'.format(path_prefix),
        model_weights='{}.weights.h5'.format(path_prefix),
        model_parameters='{}.model.yaml'.format(path_prefix),
        history='{}.history.yaml'.format(path_prefix),
        log_dir='{}.logs.d'.format(path_prefix)
        )
    return _files


def load_model(identifier):
    """Load the Recurrent Neural Network model from disk.

    Args:
        identifier: Identifier of model to load

    Returns:
        _model: RNN model

    """
    # Initialize key Variables
    _files = files(identifier)

    # Load yaml and create model
    print('> Loading model from disk')
    with open(_files.model_parameters, 'r') as yaml_file:
        loaded_model_yaml = yaml_file.read()
    _model = model_from_yaml(loaded_model_yaml)

    # Load weights into new model
    _model.load_weights(_files.model_weights, by_name=True)
    print('> Finished loading model from disk')

    # Return
    return _model


def load_history(identifier):
    """Load the Recurrent Neural Network model history from disk.

    Args:
        identifier: Identifier of model to load

    Returns:
        history: Model training history

    """
    # Initialize key Variables
    _files = files(identifier)

    # Load yaml and create model
    print('> Loading history file {} from disk'.format(_files.history))
    with open(_files.history, 'r') as yaml_file:
        history = yaml.safe_load(yaml_file)

    # Load weights into new model
    print('> Finished loading history from disk')

    # Return
    return history
