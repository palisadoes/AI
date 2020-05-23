"""Constants and other stuff we need globally."""

from collections import namedtuple

HyperParameters = namedtuple(
    'HyperParameters',
    '''units, dropout, layers, sequence_length, patience, \
batch_size, epochs, steps_per_epoch'''
)

WARMUP_STEPS = 50
