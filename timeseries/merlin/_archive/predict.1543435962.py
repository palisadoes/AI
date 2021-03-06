#!/usr/bin/env python3
"""Script to forecast data using RNN AI using GRU feedback."""

# Standard imports
import argparse
import sys
import os
import time

# PIP3 imports.
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# TensorFlow imports
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import backend

# Merlin imports
from merlin import database


class ModelVariables(object):
    """Process data for ingestion."""

    def __init__(
            self, data, batch_size=64, epochs=20):
        """Instantiate the class.

        Args:
            data: Tuple of (x_data, y_data, target_names)
            batch_size: Size of batch

        Returns:
            None

        """
        # Initialize key variables
        self._epochs = epochs
        self._batch_size = batch_size

        # Make data object accessible throughout the class
        self._data = data

        # Total number of available vectors
        num_data = len(self._data.vectors()[1])

        # Fraction of vectors to be used for training
        train_split = 0.9

        # Fraction of vectors to be used for training and testing
        self._training_count = int(train_split * num_data)

    def vectors_train(self):
        """Get vectors for learning.

        Args:
            train: Return training vectors

        Returns:
            result: Training or test vector numpy arrays

        """
        return self._vectors(train=True)

    def vectors_test(self):
        """Get vectors for validation testing.

        Args:
            None

        Returns:
            result: Training or test vector numpy arrays

        """
        return self._vectors(train=False, test_validation=True)

    def vectors_test_all(self):
        """Get vectors for testing.

        Args:
            None

        Returns:
            result: Training or test vector numpy arrays

        """
        return self._vectors(train=False, test_validation=False)

    def _vectors(self, train=True, test_validation=True):
        """Get vectors for learning.

        Note: Neither test nor training data can have NaN values for training
        to occur. We therefore have to trim the NaN values from the test data.

        Args:
            train: Return training vectors if true, else return test vectors

        Returns:
            result: Training or test vector numpy arrays

        """
        # Obtain vector data
        if train is True:
            result = self._data.vectors()[0][:self._training_count]
        else:
            if test_validation is True:
                result = self._data.vectors()[1][
                    self._training_count:-max(self._data.labels())]
            else:
                result = self._data.vectors()[1][self._training_count:]

        # Return
        return result

    def classes_train(self):
        """Get classes for training.

        Args:
            train: Return training classes

        Returns:
            result: Training or test vector numpy arrays

        """
        return self._classes(train=True)

    def classes_test(self):
        """Get classes for validation testing.

        Args:
            None

        Returns:
            result: Training or test vector numpy arrays

        """
        return self._classes(train=False)

    def _classes(self, train=True):
        """Get classes for learning.

        Note: Neither test nor training data can have NaN values for training
        to occur. We therefore have to trim the NaN values from the test data.

        NaN values occur in the vector numpy arrays. We have to make the number
        matching class rows to be the same.

        Args:
            train: Return training classes if true, else return test classes

        Returns:
            result: Training or test vector numpy arrays

        """
        # Return
        if train is True:
            result = self._data.classes()[0][:self._training_count]
        else:
            result = self._data.classes()[1][
                self._training_count:-max(self._data.labels())]
        return result

    def close(self):
        """Get closing data.

        Args:
            None

        Returns:
            result: close numpy array

        """
        # Return
        result = self._data.close()
        return result

    def datetime(self):
        """Get closing data.

        Args:
            None

        Returns:
            result: datetime numpy array

        """
        # Return
        result = self._data.datetime()
        return result

    def labels(self):
        """Get closing data.

        Args:
            None

        Returns:
            result: labels numpy array

        """
        # Return
        result = self._data.labels()
        return result

    def epochs(self):
        """Get epochs for learning.

        Args:
            None

        Returns:
            result: Number of epochs for training

        """
        # Return
        result = self._epochs
        return result

    def batch_size(self):
        """Get batch_size for learning.

        Args:
            None

        Returns:
            result: Number of batch_size for training

        """
        # Return
        result = self._batch_size
        return result

    def epoch_steps(self):
        """Get number of batch iterations to finish a training epoch.

        Args:
            None

        Returns:
            result: Number of batch_size for training

        """
        # Return
        result = int(self._training_count / self.batch_size()) + 1
        return result


class RNNGRU(object):
    """Process data for ingestion."""

    def __init__(
            self, data, sequence_length=20, warmup_steps=50, dropout=0,
            layers=1, patience=10, units=512, display=False):
        """Instantiate the class.

        Args:
            data: Tuple of (x_data, y_data, target_names)
            batch_size: Size of batch
            sequence_length: Length of vectors for for each target
            warmup_steps:

        Returns:
            None

        """
        # Initialize key variables
        self._warmup_steps = warmup_steps
        self._data = data
        self.display = display
        path_checkpoint = '/tmp/checkpoint.keras'
        _layers = int(abs(layers))

        # Delete any stale checkpoint file
        if os.path.exists(path_checkpoint) is True:
            os.remove(path_checkpoint)

        ###################################
        # TensorFlow wizardry
        config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        # Crash with DeadlineExceeded instead of hanging forever when your
        # queues get full/empty
        config.operation_timeout_in_ms = 60000

        # Create a session with the above options specified.
        backend.tensorflow_backend.set_session(tf.Session(config=config))
        ###################################

        # Get data
        self._y_current = self._data.close()

        # Create training arrays
        x_train = self._data.vectors_train()
        self._y_train = self._data.classes_train()

        # Create test arrays for VALIDATION and EVALUATION
        xv_test = self._data.vectors_test()
        self._yv_test = self._data.classes_test()

        (self.training_rows, self._training_vector_count) = x_train.shape
        (self.test_rows, _) = xv_test.shape
        (_, self._training_class_count) = self._y_train.shape

        # Print stuff
        print('\n> Numpy Data Type: {}'.format(type(x_train)))
        print("> Numpy Data Shape: {}".format(x_train.shape))
        print("> Numpy Data Row[0]: {}".format(x_train[0]))
        print("> Numpy Data Row[Last]: {}".format(x_train[-1]))
        print('> Numpy Targets Type: {}'.format(type(self._y_train)))
        print("> Numpy Targets Shape: {}".format(self._y_train.shape))

        print('> Number of Samples: {}'.format(self._y_current.shape[0]))
        print('> Number of Training Samples: {}'.format(x_train.shape[0]))
        print('> Number of Training Classes: {}'.format(
            self._training_class_count))
        print('> Number of Test Samples: {}'.format(self.test_rows))
        print("> Training Minimum Value:", np.min(x_train))
        print("> Training Maximum Value:", np.max(x_train))
        print('> Number X signals: {}'.format(self._training_vector_count))
        print('> Number Y signals: {}'.format(self._training_class_count))

        # Print epoch related data
        print('> Epochs:', self._data.epochs())
        print('> Batch Size:', self._data.batch_size())
        print('> Steps:', self._data.epoch_steps())

        # Display estimated memory footprint of training data.
        print("> Data size: {:.2f} Bytes".format(x_train.nbytes))

        '''
        The neural network works best on values roughly between -1 and 1, so we
        need to scale the data before it is being input to the neural network.
        We can use scikit-learn for this.

        We first create a scaler-object for the input-signals.

        Then we detect the range of values from the training-data and scale
        the training-data.
        '''

        self._x_scaler = MinMaxScaler()
        self._x_train_scaled = self._x_scaler.fit_transform(x_train)

        print('> Scaled Training Minimum Value: {}'.format(
            np.min(self._x_train_scaled)))
        print('> Scaled Training Maximum Value: {}'.format(
            np.max(self._x_train_scaled)))

        self._xv_test_scaled = self._x_scaler.transform(xv_test)

        '''
        The target-data comes from the same data-set as the input-signals,
        because it is the weather-data for one of the cities that is merely
        time-shifted. But the target-data could be from a different source with
        different value-ranges, so we create a separate scaler-object for the
        target-data.
        '''

        self._y_scaler = MinMaxScaler()
        self._y_train_scaled = self._y_scaler.fit_transform(self._y_train)
        yv_test_scaled = self._y_scaler.transform(self._yv_test)

        # Data Generator

        '''
        The data-set has now been prepared as 2-dimensional numpy arrays. The
        training-data has almost 300k observations, consisting of 20
        input-signals and 3 output-signals.

        These are the array-shapes of the input and output data:
        '''

        print('> Scaled Training Data Shape: {}'.format(
            self._x_train_scaled.shape))
        print('> Scaled Training Targets Shape: {}'.format(
            self._y_train_scaled.shape))

        # We then create the batch-generator.

        generator = self._batch_generator(
            self._data.batch_size(), sequence_length)

        # Validation Set

        '''
        The neural network trains quickly so we can easily run many training
        epochs. But then there is a risk of overfitting the model to the
        training-set so it does not generalize well to unseen data. We will
        therefore monitor the model's performance on the test-set after each
        epoch and only save the model's weights if the performance is improved
        on the test-set.

        The batch-generator randomly selects a batch of short sequences from
        the training-data and uses that during training. But for the
        validation-data we will instead run through the entire sequence from
        the test-set and measure the prediction accuracy on that entire
        sequence.
        '''

        validation_data = (np.expand_dims(self._xv_test_scaled, axis=0),
                           np.expand_dims(yv_test_scaled, axis=0))

        # Create the Recurrent Neural Network

        self._model = Sequential()

        '''
        We can now add a Gated Recurrent Unit (GRU) to the network. This will
        have 512 outputs for each time-step in the sequence.

        Note that because this is the first layer in the model, Keras needs to
        know the shape of its input, which is a batch of sequences of arbitrary
        length (indicated by None), where each observation has a number of
        input-signals (num_x_signals).
        '''

        self._model.add(GRU(
            units=units,
            return_sequences=True,
            recurrent_dropout=dropout,
            input_shape=(None, self._training_vector_count,)))

        for _ in range(0, _layers):
            self._model.add(GRU(
                units=units,
                recurrent_dropout=dropout,
                return_sequences=True))

        '''
        The GRU outputs a batch of sequences of 512 values. We want to predict
        3 output-signals, so we add a fully-connected (or dense) layer which
        maps 512 values down to only 3 values.

        The output-signals in the data-set have been limited to be between 0
        and 1 using a scaler-object. So we also limit the output of the neural
        network using the Sigmoid activation function, which squashes the
        output to be between 0 and 1.'''

        self._model.add(
            Dense(self._training_class_count, activation='sigmoid'))

        '''
        A problem with using the Sigmoid activation function, is that we can
        now only output values in the same range as the training-data.

        For example, if the training-data only has temperatures between -20
        and +30 degrees, then the scaler-object will map -20 to 0 and +30 to 1.
        So if we limit the output of the neural network to be between 0 and 1
        using the Sigmoid function, this can only be mapped back to temperature
        values between -20 and +30.

        We can use a linear activation function on the output instead. This
        allows for the output to take on arbitrary values. It might work with
        the standard initialization for a simple network architecture, but for
        more complicated network architectures e.g. with more layers, it might
        be necessary to initialize the weights with smaller values to avoid
        NaN values during training. You may need to experiment with this to
        get it working.
        '''

        if False:
            # Maybe use lower init-ranges.
            # init = RandomUniform(minval=-0.05, maxval=0.05)
            init = RandomUniform(minval=-0.05, maxval=0.05)

            self._model.add(Dense(
                self._training_class_count,
                activation='linear',
                kernel_initializer=init))

        # Compile Model

        '''
        This is the optimizer and the beginning learning-rate that we will use.
        We then compile the Keras model so it is ready for training.
        '''
        optimizer = RMSprop(lr=1e-3)
        self._model.compile(
            loss=self._loss_mse_warmup,
            optimizer=optimizer,
            metrics=['accuracy'])

        '''
        This is a very small model with only two layers. The output shape of
        (None, None, 3) means that the model will output a batch with an
        arbitrary number of sequences, each of which has an arbitrary number of
        observations, and each observation has 3 signals. This corresponds to
        the 3 target signals we want to predict.
        '''
        print('> Model Summary:\n')
        print(self._model.summary())

        # Callback Functions

        '''
        During training we want to save checkpoints and log the progress to
        TensorBoard so we create the appropriate callbacks for Keras.

        This is the callback for writing checkpoints during training.
        '''

        callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_weights_only=True,
                                              save_best_only=True)

        '''
        This is the callback for stopping the optimization when performance
        worsens on the validation-set.
        '''

        callback_early_stopping = EarlyStopping(monitor='val_loss',
                                                patience=patience, verbose=1)

        '''
        This is the callback for writing the TensorBoard log during training.
        '''

        callback_tensorboard = TensorBoard(log_dir='/tmp/23_logs/',
                                           histogram_freq=0,
                                           write_graph=False)

        '''
        This callback reduces the learning-rate for the optimizer if the
        validation-loss has not improved since the last epoch
        (as indicated by patience=0). The learning-rate will be reduced by
        multiplying it with the given factor. We set a start learning-rate of
        1e-3 above, so multiplying it by 0.1 gives a learning-rate of 1e-4.
        We don't want the learning-rate to go any lower than this.
        '''

        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.1,
                                               min_lr=1e-4,
                                               patience=0,
                                               verbose=1)

        callbacks = [callback_early_stopping,
                     callback_checkpoint,
                     callback_tensorboard,
                     callback_reduce_lr]

        # Train the Recurrent Neural Network

        '''We can now train the neural network.

        Note that a single "epoch" does not correspond to a single processing
        of the training-set, because of how the batch-generator randomly
        selects sub-sequences from the training-set. Instead we have selected
        steps_per_epoch so that one "epoch" is processed in a few minutes.

        With these settings, each "epoch" took about 2.5 minutes to process on
        a GTX 1070. After 14 "epochs" the optimization was stopped because the
        validation-loss had not decreased for 5 "epochs". This optimization
        took about 35 minutes to finish.

        Also note that the loss sometimes becomes NaN (not-a-number). This is
        often resolved by restarting and running the Notebook again. But it may
        also be caused by your neural network architecture, learning-rate,
        batch-size, sequence-length, etc. in which case you may have to modify
        those settings.
        '''

        print('\n> Starting data training\n')

        self._history = self._model.fit_generator(
            generator=generator,
            epochs=self._data.epochs(),
            steps_per_epoch=self._data.epoch_steps(),
            validation_data=validation_data,
            callbacks=callbacks)

        # Load Checkpoint

        '''
        Because we use early-stopping when training the model, it is possible
        that the model's performance has worsened on the test-set for several
        epochs before training was stopped. We therefore reload the last saved
        checkpoint, which should have the best performance on the test-set.
        '''

        print('> Loading model weights')
        if os.path.exists(path_checkpoint):
            self._model.load_weights(path_checkpoint)

        # Performance on Test-Set

        '''
        We can now evaluate the model's performance on the test-set. This
        function expects a batch of data, but we will just use one long
        time-series for the test-set, so we just expand the
        array-dimensionality to create a batch with that one sequence.
        '''

        result = self._model.evaluate(
            x=np.expand_dims(self._xv_test_scaled, axis=0),
            y=np.expand_dims(yv_test_scaled, axis=0))

        print('> Loss (test-set): {}'.format(result))

        # If you have several metrics you can use this instead.
        if False:
            for res, metric in zip(result, self._model.metrics_names):
                print('{0}: {1:.3e}'.format(metric, res))

    def _batch_generator(self, batch_size, sequence_length):
        """Create generator function to create random batches of training-data.

        Args:
            batch_size: Size of batch
            sequence_length: Length of sequence

        Returns:
            (x_batch, y_batch)

        """
        # Infinite loop.
        while True:
            # Allocate a new array for the batch of input-signals.
            x_shape = (
                batch_size, sequence_length, self._training_vector_count)
            x_batch = np.zeros(shape=x_shape, dtype=np.float16)

            # Allocate a new array for the batch of output-signals.
            y_shape = (batch_size, sequence_length, self._training_class_count)
            y_batch = np.zeros(shape=y_shape, dtype=np.float16)

            # Fill the batch with random sequences of data.
            for i in range(batch_size):
                # Get a random start-index.
                # This points somewhere into the training-data.
                idx = np.random.randint(
                    self.training_rows - sequence_length)

                # Copy the sequences of data starting at this index.
                x_batch[i] = self._x_train_scaled[idx:idx+sequence_length]
                y_batch[i] = self._y_train_scaled[idx:idx+sequence_length]

            yield (x_batch, y_batch)

    def _loss_mse_warmup(self, y_true, y_pred):
        """Calculate the Mean Squared Errror.

        Calculate the Mean Squared Error between y_true and y_pred,
        but ignore the beginning "warmup" part of the sequences.

        We will use Mean Squared Error (MSE) as the loss-function that will be
        minimized. This measures how closely the model's output matches the
        true output signals.

        However, at the beginning of a sequence, the model has only seen
        input-signals for a few time-steps, so its generated output may be very
        inaccurate. Using the loss-value for the early time-steps may cause the
        model to distort its later output. We therefore give the model a
        "warmup-period" of 50 time-steps where we don't use its accuracy in the
        loss-function, in hope of improving the accuracy for later time-steps

        Args:
            y_true: Desired output.
            y_pred: Model's output.

        Returns:
            loss_mean: Mean Squared Error

        """
        warmup_steps = self._warmup_steps

        # The shape of both input tensors are:
        # [batch_size, sequence_length, num_y_signals].

        # Ignore the "warmup" parts of the sequences
        # by taking slices of the tensors.
        y_true_slice = y_true[:, warmup_steps:, :]
        y_pred_slice = y_pred[:, warmup_steps:, :]

        # These sliced tensors both have this shape:
        # [batch_size, sequence_length - warmup_steps, num_y_signals]

        # Calculate the MSE loss for each value in these tensors.
        # This outputs a 3-rank tensor of the same shape.
        loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                            predictions=y_pred_slice)

        # Keras may reduce this across the first axis (the batch)
        # but the semantics are unclear, so to be sure we use
        # the loss across the entire tensor, we reduce it to a
        # single scalar with the mean function.
        loss_mean = tf.reduce_mean(loss)

        return loss_mean

    def plot_train(self, start_idx, length=100):
        """Plot the predicted and true output-signals.

        Args:
            start_idx: Start-index for the time-series.
            length: Sequence-length to process and plot.

        Returns:
            None

        """
        # Plot
        self._plot_comparison(start_idx, length=length, train=True)

    def plot_test(self, start_idx, length=100):
        """Plot the predicted and true output-signals.

        Args:
            start_idx: Start-index for the time-series.
            length: Sequence-length to process and plot.

        Returns:
            None

        """
        # Plot
        self._plot_comparison(start_idx, length=length, train=False)

    def _plot_comparison(self, start_idx, length=100, train=True):
        """Plot the predicted and true output-signals.

        Args:
            start_idx: Start-index for the time-series.
            length: Sequence-length to process and plot.
            train: Boolean whether to use training- or test-set.

        Returns:
            None

        """
        # Initialize key variables
        datetimes = {}
        num_train = self.training_rows

        # End-index for the sequences.
        end_idx = start_idx + length

        # Variables for date formatting
        days = mdates.DayLocator()   # Every day
        months = mdates.MonthLocator()  # Every month
        months_format = mdates.DateFormatter('%b %Y')
        days_format = mdates.DateFormatter('%d')

        # Assign other variables dependent on the type of data we are plotting
        if train is True:
            # Use training-data.
            x_values = self._x_train_scaled[start_idx:end_idx]
            y_true = self._y_train[start_idx:end_idx]
            shim = 'Train'

            # Datetimes to use for training
            datetimes[shim] = self._data.datetime()[
                :num_train][start_idx:end_idx]

        else:
            # Scale the data
            x_test_scaled = self._x_scaler.transform(
                self._data.vectors_test_all())

            # Use test-data.
            x_values = x_test_scaled[start_idx:end_idx]
            y_true = self._yv_test[start_idx:end_idx]
            shim = 'Test'

            # Datetimes to use for testing
            datetimes[shim] = self._data.datetime()[
                num_train:][start_idx:end_idx]

        # Input-signals for the model.
        x_values = np.expand_dims(x_values, axis=0)

        # Use the model to predict the output-signals.
        y_pred = self._model.predict(x_values)

        # The output of the model is between 0 and 1.
        # Do an inverse map to get it back to the scale
        # of the original data-set.
        y_pred_rescaled = self._y_scaler.inverse_transform(y_pred[0])

        # For each output-signal.
        for signal in range(len(self._data.labels())):
            # Assign other variables dependent on the type of data plot
            if train is True:
                # Only get current values that are a part of the training data
                current = self._y_current[:num_train][start_idx:end_idx]

                # The number of datetimes for the 'actual' plot must match
                # that of current values
                datetimes['actual'] = self._data.datetime()[
                    :num_train][start_idx:end_idx]

            else:
                # Only get current values that are a part of the test data.
                current = self._y_current[
                    num_train:][start_idx:]

                # The number of datetimes for the 'actual' plot must match
                # that of current values
                datetimes['actual'] = self._data.datetime()[
                    num_train:][start_idx:]

            # Create a filename
            filename = (
                '/tmp/batch_{}_epochs_{}_training_{}_{}_{}_{}.png').format(
                    self._data.batch_size(),
                    self._data.epochs(),
                    num_train,
                    signal,
                    int(time.time()),
                    shim)

            # Get the output-signal predicted by the model.
            signal_pred = y_pred_rescaled[:, signal]

            # Get the true output-signal from the data-set.
            signal_true = y_true[:, signal]

            # Create a new chart
            (fig, axis) = plt.subplots(figsize=(15, 5))

            # Plot and compare the two signals.
            axis.plot(
                datetimes[shim][:len(signal_true)],
                signal_true,
                label='Current +{}'.format(self._data.labels()[signal]))
            axis.plot(
                datetimes[shim][:len(signal_pred)],
                signal_pred,
                label='Prediction')
            axis.plot(datetimes['actual'], current, label='Current')

            # Set plot labels and titles
            axis.set_title('{1}ing Forecast ({0} Future Intervals)'.format(
                self._data.labels()[signal], shim))
            axis.set_ylabel('Values')
            axis.legend(
                bbox_to_anchor=(1.04, 0.5),
                loc='center left', borderaxespad=0)

            # Add gridlines and ticks
            ax = plt.gca()
            ax.grid(True)

            # Add major gridlines
            ax.xaxis.grid(which='major', color='black', alpha=0.2)
            ax.yaxis.grid(which='major', color='black', alpha=0.2)

            # Add minor ticks (They must be turned on first)
            ax.minorticks_on()
            ax.xaxis.grid(which='minor', color='black', alpha=0.1)
            ax.yaxis.grid(which='minor', color='black', alpha=0.1)

            # Format the tick labels
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(months_format)
            ax.xaxis.set_minor_locator(days)

            # Remove tick marks
            ax.tick_params(axis='both', which='both', length=0)

            # Print day numbers on xaxis for Test data only
            if train is False:
                ax.xaxis.set_minor_formatter(days_format)
                plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)

            # Rotates and right aligns the x labels, and moves the bottom of
            # the axes up to make room for them
            fig.autofmt_xdate()

            # Plot grey box for warmup-period if we are working with training
            # data and the start is within the warmup-period
            if (0 < start_idx < self._warmup_steps):
                if train is True:
                    plt.axvspan(
                        datetimes[shim][start_idx],
                        datetimes[shim][self._warmup_steps],
                        facecolor='black', alpha=0.15)

            # Show and save the image
            if self.display is True:
                fig.savefig(filename, bbox_inches='tight')
                plt.show()
            else:
                fig.savefig(filename, bbox_inches='tight')
            print('> Saving file: {}'.format(filename))

            # Close figure
            plt.close(fig=fig)

    def plot_accuracy(self):
        """Plot the predicted and true output-signals.

        Args:
            None

        Returns:
            None

        """
        # Summarize history for accuracy
        plt.figure(figsize=(15, 5))
        plt.plot(self._history.history['acc'])
        plt.plot(self._history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # Summarize history for loss
        plt.figure(figsize=(15, 5))
        plt.plot(self._history.history['loss'])
        plt.plot(self._history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


def main():
    """Generate forecasts.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    periods = 1
    ts_start = int(time.time())

    # Set logging level - No Tensor flow messages
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Get CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='Name of CSV file to read.',
        type=str, required=True)
    parser.add_argument(
        '-b', '--batch-size', help='Size of batch. Default 500.',
        type=int, default=500)
    parser.add_argument(
        '-w', '--weeks',
        help='Number of weeks of data to consider when learning. Default 53.',
        type=int, default=53)
    parser.add_argument(
        '-e', '--epochs',
        help='Number of epoch iterations to use. Default 20.',
        type=int, default=20)
    parser.add_argument(
        '-p', '--patience',
        help=(
            'Number of iterations of without loss improvement '
            'before exiting.'),
        type=int, default=10)
    parser.add_argument(
        '-l', '--layers',
        help='Number of GRU layers to use.',
        type=int, default=2)
    parser.add_argument(
        '-d', '--dropout',
        help='Dropout rate as decimal from 0 to 1. Default 0.5 (or 50%)',
        type=float, default=0.5)
    parser.add_argument(
        '-u', '--units',
        help='Number of units per layer. Default 512',
        type=int, default=512)
    parser.add_argument(
        '--display',
        help='Display on screen if True. Default False.',
        action='store_true')
    args = parser.parse_args()
    filename = args.filename
    display = args.display
    dropout = args.dropout
    layers = max(0, args.layers - 1)
    patience = args.patience

    '''
    We will use a large batch-size so as to keep the GPU near 100% work-load.
    You may have to adjust this number depending on your GPU, its RAM and your
    choice of sequence_length below.

    Batch size is the number of samples per gradient update. It is a set of N
    samples. The samples in a batch are processed independently, in parallel.
    If training, a batch results in only one update to the model.

    In other words, "batch size" is the number of samples you put into for each
    training round to calculate the gradient. A training round would be a
    "step" within an epoch.

    A batch generally approximates the distribution of the input data better
    than a single input. The larger the batch, the better the approximation;
    however, it is also true that the batch will take longer to process and
    will still result in only one update. For inference (evaluate/predict),
    it is recommended to pick a batch size that is as large as you can afford
    without going out of memory (since larger batches will usually result in
    faster evaluating/prediction).
    '''

    batch_size = args.batch_size

    '''
    We will use a sequence-length of 1344, which means that each random
    sequence contains observations for 8 weeks. One time-step corresponds to
    one hour, so 24 x 7 time-steps corresponds to a week, and 24 x 7 x 8
    corresponds to 8 weeks.
    '''
    weeks = args.weeks
    sequence_length = 5 * periods * weeks

    '''
    An epoch is an arbitrary cutoff, generally defined as "one pass over the
    entire dataset", used to separate training into distinct phases, which is
    useful for logging and periodic evaluation.

    Number of epochs to train the model. An epoch is an iteration over the
    entire x and y data provided. Note that in conjunction with initial_epoch,
    epochs is to be understood as "final epoch". The model is not trained for a
    number of iterations given by epochs, but merely until the epoch of index
    epochs is reached.
    '''
    epochs = args.epochs

    # Get the data
    lookahead_periods = [1, 3]
    training_data = database.DataGRU(filename, lookahead_periods)
    data = ModelVariables(training_data, batch_size=batch_size, epochs=epochs)

    # Do training
    rnn = RNNGRU(
        data,
        sequence_length=sequence_length,
        dropout=dropout,
        layers=layers,
        patience=patience,
        display=display)

    '''
    Calculate the duration
    '''

    duration = int(time.time()) - ts_start

    '''
    We can now plot an example of predicted output-signals. It is important to
    understand what these plots show, as they are actually a bit more
    complicated than you might think.

    These plots only show the output-signals and not the 20 input-signals used
    to predict the output-signals. The time-shift between the input-signals
    and the output-signals is held fixed in these plots. The model always
    predicts the output-signals e.g. 24 hours into the future (as defined in
    the shift_steps variable above). So the plot's x-axis merely shows how many
    time-steps of the input-signals have been seen by the predictive model so
    far.

    The prediction is not very accurate for the first 30-50 time-steps because
    the model has seen very little input-data at this point. The model
    generates a single time-step of output data for each time-step of the
    input-data, so when the model has only run for a few time-steps, it knows
    very little of the history of the input-signals and cannot make an accurate
    prediction. The model needs to "warm up" by processing perhaps 30-50
    time-steps before its predicted output-signals can be used.

    That is why we ignore this "warmup-period" of 50 time-steps when
    calculating the mean-squared-error in the loss-function. The
    "warmup-period" is shown as a grey box in these plots.

    Let us start with an example from the training-data. This is data that the
    model has seen during training so it should perform reasonably well on
    this data.

    NOTE:

    The model was able to predict the overall oscillations of the temperature
    quite well but the peaks were sometimes inaccurate. For the wind-speed, the
    overall oscillations are predicted reasonably well but the peaks are quite
    inaccurate. For the atmospheric pressure, the overall curve-shape has been
    predicted although there seems to be a slight lag and the predicted curve
    has a lot of noise compared to the smoothness of the original signal.

    '''

    rnn.plot_train(start_idx=rnn.training_rows - 250, length=250)

    # Example from Test-Set

    '''
    Now consider an example from the test-set. The model has not seen this data
    during training.

    The temperature is predicted reasonably well, although the peaks are
    sometimes inaccurate.

    The wind-speed has not been predicted so well. The daily
    oscillation-frequency seems to match, but the center-level and the peaks
    are quite inaccurate. A guess would be that the wind-speed is difficult to
    predict from the given input data, so the model has merely learnt to output
    sinusoidal oscillations in the daily frequency and approximately at the
    right center-level.

    The atmospheric pressure is predicted reasonably well, except for a lag and
    a more noisy signal than the true time-series.
    '''

    rnn.plot_test(start_idx=rnn.test_rows-30, length=rnn.test_rows)

    # Plot accuracy
    # rnn.plot_accuracy()

    # Print duration
    print("> Training Duration: {}s".format(duration))


if __name__ == "__main__":
    main()
