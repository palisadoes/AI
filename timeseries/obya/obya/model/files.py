"""File manipulation module."""

from collections import namedtuple


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
        model_parameters='{}.model'.format(path_prefix),
        history='{}.history.yaml'.format(path_prefix),
        log_dir='{}.logs.d'.format(path_prefix)
        )
    return _files
