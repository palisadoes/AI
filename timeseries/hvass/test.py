#!/usr/bin/env python3
"""Script to forecast timeseries data."""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean

import weather


def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.

    y_true is the desired output.
    y_pred is the model's output.
    """
    warmup_steps = 50

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculat the Mean Squared Error and use it as loss.
    mse = mean(square(y_true_slice - y_pred_slice))

    return mse


def batch_generator(
        batch_size, sequence_length, num_x_signals, num_y_signals,
        num_train, x_train_scaled, y_train_scaled):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]

        yield (x_batch, y_batch)


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    weather.maybe_download_and_extract()

    cities = weather.cities
    df = weather.load_resampled_data()

    df.drop(('Esbjerg', 'Pressure'), axis=1, inplace=True)
    df.drop(('Roskilde', 'Pressure'), axis=1, inplace=True)

    df['Various', 'Day'] = df.index.dayofyear
    df['Various', 'Hour'] = df.index.hour

    target_city = 'Odense'
    target_names = ['Temp', 'WindSpeed', 'Pressure']

    shift_days = 1
    shift_steps = shift_days * 2

    df_targets = df[target_city][target_names].shift(-shift_steps)
    x_data = df.values[0:-shift_steps]

    print(type(x_data))
    print("Shape:", x_data.shape)

    y_data = df_targets.values[:-shift_steps]

    num_data = len(x_data)

    train_split = 0.9

    num_train = int(train_split * num_data)
    num_test = num_data - num_train

    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]
    print(len(x_train) + len(x_test))

    y_train = y_data[0:num_train]
    y_test = y_data[num_train:]
    print(len(y_train) + len(y_test))

    num_x_signals = x_data.shape[1]
    num_y_signals = y_data.shape[1]

    print("Min:", np.min(x_train))
    print("Max:", np.max(x_train))

    x_scaler = MinMaxScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)

    print("Min:", np.min(x_train_scaled))
    print("Max:", np.max(x_train_scaled))

    x_test_scaled = x_scaler.transform(x_test)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    print(x_train_scaled.shape)
    print(y_train_scaled.shape)

    batch_size = 64
    sequence_length = 24 * 7 * 8
    generator = batch_generator(
        batch_size, sequence_length, num_x_signals, num_y_signals, num_train,
        x_train_scaled, y_train_scaled)

    x_batch, y_batch = next(generator)

    print(x_batch.shape)
    print(y_batch.shape)

    validation_data = (np.expand_dims(x_test_scaled, axis=0),
                       np.expand_dims(y_test_scaled, axis=0))

    with tf.device('/:GPU:0'):
        model = Sequential()
    model.add(GRU(units=512,
                  return_sequences=True,
                  input_shape=(None, num_x_signals,)))

    model.add(Dense(num_y_signals, activation='sigmoid'))

    if False:
        from tensorflow.python.keras.initializers import RandomUniform

        # Maybe use lower init-ranges.
        init = RandomUniform(minval=-0.05, maxval=0.05)

        model.add(Dense(num_y_signals,
                        activation='linear',
                        kernel_initializer=init))

    optimizer = RMSprop(lr=1e-3)

    model.compile(loss=loss_mse_warmup, optimizer=optimizer)
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

    model.fit(x=generator,
              epochs=20,
              steps_per_epoch=100,
              validation_data=validation_data,
              callbacks=callbacks)


if __name__ == "__main__":
    main()
