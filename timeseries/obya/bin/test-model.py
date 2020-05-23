#!/usr/bin/env python3
"""Script to forecast timeseries data."""

import time
from collections import namedtuple
import numpy as np
import pandas as pd
import argparse
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean

from obya.etl import etl
from obya.model import memory


def main():
    args = arguments()
    df_ = pd.read_csv(args.filename, names=['timestamp', 'value'], index_col=0)
    data = etl.Data(df_)
    model = Model(data)
    model.model()


class Model():
    """Process data for ingestion.

    Roughly based on:

    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb

    """

    def __init__(
            self, _data, batch_size=256, epochs=20,
            sequence_length=1500, dropout=0.1,
            layers=1, patience=5, units=128, multigpu=False):
        """Instantiate the class.

        Args:
            data: etl.Data object
            batch_size: Size of batch
            sequence_length: Length of vectors for for each target
            warmup_steps:
            display: Show charts of results if True

        Returns:
            None

        """
        # Setup memory
        self._processors = memory.setup()

        # Initialize key variables
        if multigpu is True:
            self._gpus = len(self._processors.gpus)
        else:
            self._gpus = 1
        self._batch_size = batch_size * self._gpus
        self._sequence_length = sequence_length

        # Set key file locations
        path_prefix = '/tmp/hvass-{}'.format(int(time.time()))
        Files = namedtuple(
            'Files', 'checkpoint, model_weights, model_parameters')
        self._files = Files(
            checkpoint='{}.checkpoint.h5'.format(path_prefix),
            model_weights='{}.weights.h5'.format(path_prefix),
            model_parameters='{}.model.yaml'.format(path_prefix)
            )

        # Initialize parameters
        HyperParameters = namedtuple(
            'HyperParameters',
            '''units, dropout, layers, sequence_length, patience, \
batch_size, epochs'''
        )
        self._hyperparameters = HyperParameters(
            units=abs(units),
            dropout=abs(dropout),
            layers=int(abs(layers)),
            sequence_length=abs(sequence_length),
            patience=abs(patience),
            batch_size=int(self._batch_size),
            epochs=abs(epochs)
        )

        # Delete any stale checkpoint file
        if os.path.exists(self._files.checkpoint) is True:
            os.remove(self._files.checkpoint)

        # Get data
        self._data = _data

    def model(self, params=None):

        normal = self._data.split()
        scaled = self._data.scaled_split()
        (training_rows, x_feature_count) = normal.x_train.shape
        (_, y_feature_count) = normal.y_train.shape

        # Allow overriding parameters
        if params is None:
            _hyperparameters = self._hyperparameters
        else:
            _hyperparameters = params
            _hyperparameters.batch_size = int(
                _hyperparameters.batch_size * self._gpus)

        Generator = namedtuple(
            'Generator',
            '''batch_size, sequence_length, x_feature_count, y_feature_count, \
training_rows, y_train_scaled, x_train_scaled''')
        generator = _batch_generator(Generator(
            batch_size=_hyperparameters.batch_size,
            sequence_length=_hyperparameters.sequence_length,
            x_feature_count=x_feature_count,
            y_feature_count=y_feature_count,
            training_rows=training_rows,
            y_train_scaled=scaled.y_train,
            x_train_scaled=scaled.x_train
        ))

        validation_data = (
            np.expand_dims(scaled.x_train, axis=0),
            np.expand_dims(scaled.y_train, axis=0)
        )

        with tf.device('/:GPU:0'):
            model = Sequential()

        model.add(GRU(
            _hyperparameters.units,
            return_sequences=True,
            recurrent_dropout=_hyperparameters.dropout,
            input_shape=(None, x_feature_count)))

        model.add(Dense(y_feature_count, activation='sigmoid'))

        if False:
            from tensorflow.python.keras.initializers import RandomUniform

            # Maybe use lower init-ranges.
            init = RandomUniform(minval=-0.05, maxval=0.05)

            model.add(Dense(y_feature_count,
                            activation='linear',
                            kernel_initializer=init))

        optimizer = RMSprop(lr=1e-3)
        model.compile(
            loss=self.loss_mse_warmup,
            optimizer=optimizer,
            metrics=['accuracy'])
        model.summary()

        path_checkpoint = '/tmp/hvass_checkpoint.keras'
        callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_weights_only=True,
                                              save_best_only=True)

        callback_early_stopping = EarlyStopping(monitor='val_loss',
                                                patience=5, verbose=1)

        callback_tensorboard = TensorBoard(log_dir='/tmp/hvass_logs/',
                                           histogram_freq=0,
                                           write_graph=False)

        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.1,
                                               min_lr=1e-4,
                                               patience=0,
                                               verbose=1)

        callbacks = [callback_early_stopping,
                     callback_checkpoint,
                     callback_tensorboard,
                     callback_reduce_lr]

        model.fit(
            x=generator,
            epochs=20,
            steps_per_epoch=100,
            validation_data=validation_data,
            callbacks=callbacks)

    def loss_mse_warmup(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error between y_true and y_pred,
        but ignore the beginning "warmup" part of the sequences.

        y_true is the desired output.
        y_pred is the model's output.
        """
        warmup_steps = 50

        # The shape of both input tensors are:
        # [batch_size, sequence_length, y_feature_count].

        # Ignore the "warmup" parts of the sequences
        # by taking slices of the tensors.
        y_true_slice = y_true[:, warmup_steps:, :]
        y_pred_slice = y_pred[:, warmup_steps:, :]

        # These sliced tensors both have this shape:
        # [batch_size, sequence_length - warmup_steps, y_feature_count]

        # Calculat the Mean Squared Error and use it as loss.
        mse = mean(square(y_true_slice - y_pred_slice))

        return mse


def _batch_generator(parameters):
    """
    Generator function for creating random batches of training-data.
    """
    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (
            parameters.batch_size,
            parameters.sequence_length,
            parameters.x_feature_count
        )
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (
            parameters.batch_size,
            parameters.sequence_length,
            parameters.y_feature_count
        )
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(parameters.batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(
                parameters.training_rows - parameters.sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = parameters.x_train_scaled[
                idx:idx+parameters.sequence_length]
            y_batch[i] = parameters.y_train_scaled[
                idx:idx+parameters.sequence_length]

        yield (x_batch, y_batch)


def arguments():
    """Get the CLI arguments.

    Args:
        None

    Returns:
        args: NamedTuple of argument values

    """
    # Get config_dir value
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename',
        help=('Name of file to process.'),
        required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
