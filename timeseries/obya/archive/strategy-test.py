#!/usr/bin/env python3
"""Script to forecast timeseries data.

Baased on:

    https://github.com/tensorflow/tensorflow/issues/34067#issuecomment-559109872

"""

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

NUM_WORKERS = 2

data = np.random.uniform(-1.0, 1.0, (10000, 100, 150, 3))
labels = np.random.uniform(-1.0, 1.0, (10000, 10))


def get_model():
    inputs = keras.Input(shape=(100, 150, 3), name='rgb_input')

    net = layers.Conv2D(16, 9, 2, 'same', activation='relu')(inputs)
    net = layers.Flatten()(net)
    net = layers.Dense(10, activation=K.tanh)(net)

    return keras.Model(inputs=inputs, outputs=net)


def _callbacks():
    """Create callbacks for learning.

    Args:
        _files: model.files.Files tuple of data

    Returns:

        callbacks: List of callbacks

    """

    '''
    During training we want to save checkpoints and log the progress to
    TensorBoard so we create the appropriate callbacks for Keras.

    This is the callback for writing checkpoints during training.
    '''

    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='/tmp/_files.checkpoint.h5', monitor='val_loss',
        verbose=1, save_weights_only=True, save_best_only=True
    )

    '''
    This is the callback for stopping the optimization when performance
    worsens on the validation-set.
    '''

    callback_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, verbose=1
    )

    '''
    This is the callback for writing the TensorBoard log during training.
    '''

    callback_tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='/tmp/_files.log_dir', histogram_freq=0, write_graph=False
    )

    '''
    This callback reduces the learning-rate for the optimizer if the
    validation-loss has not improved since the last epoch
    (as indicated by patience=0). The learning-rate will be reduced by
    multiplying it with the given factor. We set a start learning-rate of
    1e-3 above, so multiplying it by 0.1 gives a learning-rate of 1e-4.
    We don't want the learning-rate to go any lower than this.
    '''

    callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, min_lr=1e-4, patience=0, verbose=1
    )

    callbacks = [callback_early_stopping,
                 callback_checkpoint,
                 callback_tensorboard,
                 callback_reduce_lr]

    return callbacks


def main():
    """Process data.

    Display data prediction from tensorflow model

    """
    # simple early stopping
    callbacks = _callbacks()

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2)

    validation_data = (x_test, y_test)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        model = get_model()
        model.compile(
            optimizer=keras.optimizers.Adam(3e-4),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()])

    model.fit(
        x_train, y_train, batch_size=128 * NUM_WORKERS,
        validation_data=validation_data,
        epochs=25, callbacks=[callbacks])


if __name__ == "__main__":
    main()
