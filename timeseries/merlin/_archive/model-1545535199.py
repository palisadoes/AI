#!/usr/bin/env python3
"""Script to forecast data using RNN AI using GRU feedback."""

# Standard imports
from __future__ import print_function
import time
import os
import sys
from copy import deepcopy
from pprint import pprint
import gc
import inspect

# PIP3 imports.
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hyperopt import STATUS_OK
from statsmodels.tsa.stattools import adfuller

# TensorFlow imports
import tensorflow as tf

# Keras imports
from keras.models import Sequential, model_from_json
from keras.layers import Dense, GRU
from keras.optimizers import RMSprop
from keras.initializers import RandomUniform
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import backend
from keras.utils import multi_gpu_model
from keras import Model


# Merlin imports
from merlin import general


class RNNGRU(object):
    """Process data for ingestion.

    Roughly based on:

    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb

    """

    def __init__(
            self, _data, batch_size=64, epochs=20,
            sequence_length=20, warmup_steps=50, dropout=0,
            layers=1, patience=10, units=512, display=False, binary=False):
        """Instantiate the class.

        Args:
            data: Tuple of (x_data, y_data, target_names)
            batch_size: Size of batch
            sequence_length: Length of vectors for for each target
            warmup_steps:
            display: Show charts of results if True
            binary: Process data for predicting boolean up / down movement vs

                actual values if True
        Returns:
            None

        """
        # Initialize key variables
        self._warmup_steps = warmup_steps
        self._binary = binary
        self._display = display
        self._data = _data

        # Set key file locations
        path_prefix = '/tmp/keras-{}'.format(int(time.time()))
        self._path_checkpoint = '{}.checkpoint'.format(path_prefix)
        self._path_model_weights = '{}.weights.h5'.format(path_prefix)
        self._path_model_parameters = '{}.model'.format(path_prefix)

        # Initialize parameters
        self.hyperparameters = {
            'units': units,
            'dropout': dropout,
            'layers': int(abs(layers)),
            'sequence_length': sequence_length,
            'patience': patience,
            'batch_size': batch_size,
            'epochs': epochs
        }

        # Delete any stale checkpoint file
        if os.path.exists(self._path_checkpoint) is True:
            os.remove(self._path_checkpoint)

        ###################################
        # TensorFlow wizardry
        config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        # Crash with DeadlineExceeded instead of hanging forever when your
        # queues get full/empty
        config.operation_timeout_in_ms = 60000

        # Create a session with the above options specified.
        backend.tensorflow_backend.set_session(tf.Session(config=config))
        ###################################

        # Get data
        self._y_current = self._data.values()

        # Create test and training arrays for VALIDATION and EVALUATION
        (x_train,
         x_validation,
         _x_test,
         self._y_train,
         self._y_validation,
         self._y_test) = self._data.train_validation_test_split()

        (self.training_rows, self._training_vector_count) = x_train.shape
        (self.test_rows, _) = _x_test.shape
        (_, self._training_class_count) = self._y_train.shape

        '''
        The neural network works best on values roughly between -1 and 1, so we
        need to scale the data before it is being input to the neural network.
        We can use scikit-learn for this.

        We first create a scaler-object for the input-signals.

        Then we detect the range of values from the training-data and scale
        the training-data.

        From StackOverflow:

        To center the data (make it have zero mean and unit standard error),
        you subtract the mean and then divide the result by the standard
        deviation.

            x'=x−μσ

        You do that on the training set of data. But then you have to apply the
        same transformation to your testing set (e.g. in cross-validation), or
        to newly obtained examples before forecast. But you have to use the
        same two parameters μ and σ (values) that you used for centering the
        training set.

        Hence, every sklearn's transform's fit() just calculates the parameters
        (e.g. μ and σ in case of StandardScaler) and saves them as an internal
        objects state. Afterwards, you can call its transform() method to apply
        the transformation to a particular set of examples.

        fit_transform() joins these two steps and is used for the initial
        fitting of parameters on the training set x, but it also returns a
        transformed x'. Internally, it just calls first fit() and then
        transform() on the same data.
        '''
        self._x_scaler = MinMaxScaler()
        _ = self._x_scaler.fit_transform(self._data.vectors())
        self._x_train_scaled = self._x_scaler.transform(x_train)
        self._x_validation_scaled = self._x_scaler.transform(x_validation)
        self._x_test_scaled = self._x_scaler.transform(_x_test)

        '''print(np.amin(self._x_train_scaled), np.amax(self._x_train_scaled))
        print(np.amin(self._x_train_scaled), np.amax(self._x_train_scaled))
        print(np.amin(self._x_test_scaled), np.amax(self._x_test_scaled))

        print('\n', _x_test, '\n')
        print('\n', self._x_test_scaled, '\n')'''

        '''
        The target-data comes from the same data-set as the input-signals,
        because it is the weather-data for one of the cities that is merely
        time-shifted. But the target-data could be from a different source with
        different value-ranges, so we create a separate scaler-object for the
        target-data.
        '''

        self._y_scaler = MinMaxScaler()
        _ = self._y_scaler.fit_transform(self._data.classes())
        self._y_train_scaled = self._y_scaler.transform(self._y_train)
        self._y_validation_scaled = self._y_scaler.transform(
            self._y_validation)
        self._y_test_scaled = self._y_scaler.transform(self._y_test)

        '''print(np.amin(self._y_train_scaled), np.amax(self._y_train_scaled))
        print(np.amin(self._y_train_scaled), np.amax(self._y_train_scaled))
        print(np.amin(self._y_test_scaled), np.amax(self._y_test_scaled))
        sys.exit()'''

        # Print stuff
        print('\n> Numpy Data Type: {}'.format(type(x_train)))
        print("> Numpy Data Shape: {}".format(x_train.shape))
        print("> Numpy Data Row[0]: {}".format(x_train[0]))
        print("> Numpy Data Row[Last]: {}".format(x_train[-1]))
        print('> Numpy Targets Type: {}'.format(type(self._y_train)))
        print("> Numpy Vector Feature Type: {}".format(type(x_train[0][0])))
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
        print('> Epochs:', self.hyperparameters['epochs'])
        print('> Batch Size:', self.hyperparameters['batch_size'])

        # Display estimated memory footprint of training data.
        print("> Data size: {:.2f} Bytes".format(x_train.nbytes))

        print('> Scaled Training Minimum Value: {}'.format(
            np.min(self._x_train_scaled)))
        print('> Scaled Training Maximum Value: {}'.format(
            np.max(self._x_train_scaled)))

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

    def model(self, params=None):
        """Create the Recurrent Neural Network.

        Args:
            None

        Returns:
            _model: RNN model

        """
        # Initialize key variables
        gpus = len(general.get_available_gpus())
        if params is None:
            _hyperparameters = self.hyperparameters
        else:
            _hyperparameters = params

        # Calculate the steps per epoch
        epoch_steps = int(
            self.training_rows / _hyperparameters['batch_size']) + 1

        '''
        Instantiate the base model (or "template" model).
        We recommend doing this with under a CPU device scope,
        so that the model's weights are hosted on CPU memory.
        Otherwise they may end up hosted on a GPU, which would
        complicate weight sharing.

        NOTE: multi_gpu_model values will be way off if you don't do this.
        '''
        with tf.device('/cpu:0'):
            serial_model = Sequential()

        '''
        We can now add a Gated Recurrent Unit (GRU) to the network. This will
        have 512 outputs for each time-step in the sequence.

        Note that because this is the first layer in the model, Keras needs to
        know the shape of its input, which is a batch of sequences of arbitrary
        length (indicated by None), where each observation has a number of
        input-signals (num_x_signals).
        '''

        serial_model.add(GRU(
            _hyperparameters['units'],
            return_sequences=True,
            recurrent_dropout=_hyperparameters['dropout'],
            input_shape=(None, self._training_vector_count,)))

        for _ in range(1, _hyperparameters['layers']):
            serial_model.add(GRU(
                _hyperparameters['units'],
                recurrent_dropout=_hyperparameters['dropout'],
                return_sequences=True))

        '''
        The GRU outputs a batch from keras_contrib.layers.advanced_activations
        of sequences of 512 values. We want to predict
        3 output-signals, so we add a fully-connected (or dense) layer which
        maps 512 values down to only 3 values.

        The output-signals in the data-set have been limited to be between 0
        and 1 using a scaler-object. So we also limit the output of the neural
        network using the Sigmoid activation function, which squashes the
        output to be between 0 and 1.
        '''

        serial_model.add(
            Dense(self._training_class_count, activation='sigmoid'))

        '''
        A problem with using the Sigmoid activation function, is that we can
        now only output values in the same range as the training-data.

        For example, if the training-data only has values between -20 and +30,
        then the scaler-object will map -20 to 0 and +30 to 1. So if we limit
        the output of the neural network to be between 0 and 1 using the
        Sigmoid function, this can only be mapped back to values between
        -20 and +30.

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
            init = RandomUniform(minval=-0.05, maxval=0.05)

            serial_model.add(Dense(
                self._training_class_count,
                activation='linear',
                kernel_initializer=init))

        '''print(inspect.getmembers(_model, predicate=inspect.ismethod))
        print('\n\n----------------------\n\n')'''

        # Apply multi-GPU logic.
        try:
            # We have to wrap multi_gpu_model this way to get callbacks to work
            _model = ModelMGPU(
                serial_model,
                cpu_relocation=True,
                gpus=gpus)
            '''_model = multi_gpu_model(
                _model,
                cpu_relocation=True,
                gpus=gpus)'''
            #_model = serial_model
            print('> Training using multiple GPUs...')
        except ValueError:
            print('> Training using single GPU or CPU...')

        '''print(inspect.getmembers(_model, predicate=inspect.ismethod))
        sys.exit(0)'''

        # Compile Model

        '''
        This is the optimizer and the beginning learning-rate that we will use.
        We then compile the Keras model so it is ready for training.
        '''

        optimizer = RMSprop(lr=1e-3)
        if self._binary is True:
            _model.compile(
                loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
        else:
            _model.compile(
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
        print('\n> Model Summary:\n')
        print(_model.summary())

        # Create the batch-generator.
        generator = self._batch_generator(
            _hyperparameters['batch_size'],
            _hyperparameters['sequence_length'])

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

        validation_data = (np.expand_dims(self._x_validation_scaled, axis=0),
                           np.expand_dims(self._y_validation_scaled, axis=0))

        # Callback Functions

        '''
        During training we want to save checkpoints and log the progress to
        TensorBoard so we create the appropriate callbacks for Keras.

        This is the callback for writing checkpoints during training.
        '''

        callback_checkpoint = ModelCheckpoint(filepath=self._path_checkpoint,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_weights_only=True,
                                              save_best_only=True)

        '''
        This is the callback for stopping the optimization when performance
        worsens on the validation-set.
        '''

        callback_early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=_hyperparameters['patience'],
            verbose=1)

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

        print('\n> Parameters for training\n')
        pprint(_hyperparameters)
        print('\n> Starting data training\n')

        _model.fit_generator(
            generator=generator,
            epochs=_hyperparameters['epochs'],
            steps_per_epoch=epoch_steps,
            use_multiprocessing=True,
            validation_data=validation_data,
            callbacks=callbacks)

        # Return
        return serial_model

    def save(self, _model):
        """Save the Recurrent Neural Network model.

        Args:
            None

        Returns:
            _model: RNN model

        """
        # Serialize model to JSON
        model_json = _model.to_json()
        with open(self._path_model_parameters, 'w') as json_file:
            json_file.write(model_json)

        # Serialize weights to HDF5
        _model.save_weights(self._path_model_weights)
        print('> Saved model to disk')

    def load_model(self):
        """Load the Recurrent Neural Network model from disk.

        Args:
            None

        Returns:
            _model: RNN model

        """
        # Load json and create model
        print('> Loading model from disk')
        with open(self._path_model_parameters, 'r') as json_file:
            loaded_model_json = json_file.read()
        _model = model_from_json(loaded_model_json)

        # Load weights into new model
        _model.load_weights(self._path_model_weights)
        '''sys.exit(0)
        _model.load_weights(self._path_checkpoint)'''
        print('> Finished loading model from disk')

        # Return
        return _model

    def evaluate(self):
        """Evaluate the model.

        Args:
            _model: Model to evaluate

        Returns:
            None

        """
        # Load Checkpoint

        '''
        Because we use early-stopping when training the model, it is possible
        that the model's performance has worsened on the test-set for several
        epochs before training was stopped. We therefore reload the last saved
        checkpoint, which should have the best performance on the test-set.
        '''

        '''if os.path.exists(self._path_checkpoint):
            _model.load_weights(self._path_checkpoint)'''

        _model = self.load_model()

        optimizer = RMSprop(lr=1e-3)
        if self._binary is True:
            _model.compile(
                loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
        else:
            _model.compile(
                loss=self._loss_mse_warmup,
                optimizer=optimizer,
                metrics=['accuracy'])

        # Performance on Test-Set

        '''
        We can now evaluate the model's performance on the validation-set.
        This function expects a batch of data, but we will just use one long
        time-series for the test-set, so we just expand the
        array-dimensionality to create a batch with that one sequence.
        '''

        if self._binary is False:
            x_scaled = self._x_test_scaled
            y_scaled = self._y_test_scaled
        else:
            # Get the filtered vectors and classes
            (filtered_vectors,
             filtered_classes) = self._data.stochastic_vectors_classes()

            # Scale and then evaluate
            x_scaled = self._x_scaler.transform(filtered_vectors)
            y_scaled = self._y_scaler.transform(filtered_classes)

        # Evaluate the MSE accuracy
        result = _model.evaluate(
            x=np.expand_dims(x_scaled, axis=0),
            y=np.expand_dims(y_scaled, axis=0))

        # If you have several metrics you can use this instead.
        print('> Metrics (test-set):')
        for _value, _metric in zip(result, _model.metrics_names):
            print('\t{}: {:.10f}'.format(_metric, _value))

        if self._binary is True:
            # Input-signals for the model.
            x_values = np.expand_dims(x_scaled, axis=0)

            # Get the predictions
            predictions_scaled = _model.predict_classes(x_values, verbose=1)

            # The output of the model is between 0 and 1.
            # Do an inverse map to get it back to the scale
            # of the original data-set.
            predictions = self._y_scaler.inverse_transform(
                predictions_scaled[0])

            # Print meaningful human accuracy values
            print(
                '> Human accuracy {:.3f} %'
                ''.format(general.binary_accuracy(
                    predictions, filtered_classes) * 100))

    def objective(self, params=None):
        """Optimize the Recurrent Neural Network.

        Args:
            None

        Returns:
            _model: RNN model

        """
        model = deepcopy(self.model(params=params))

        if bool(self._binary) is False:
            scaled_vectors = self._x_test_scaled
            test_classes = self._y_test
        else:
            # Get the filtered vectors and classes
            (filtered_vectors,
             test_classes) = self._data.stochastic_vectors_classes()

            # Scale and then evaluate
            scaled_vectors = self._x_scaler.transform(filtered_vectors)

        # Input-signals for the model.
        x_values = np.expand_dims(scaled_vectors, axis=0)

        # Get the predictions
        predictions_scaled = model.predict(x_values, verbose=1)

        # The output of the model is between 0 and 1.
        # Do an inverse map to get it back to the scale
        # of the original data-set.
        predictions = self._y_scaler.inverse_transform(
            predictions_scaled[0])

        # Get the error value
        accuracy = mean_absolute_error(test_classes, predictions)

        # Free object memory
        del model
        gc.collect()

        # Print meaningful human accuracy values
        if self._binary is True:
            # Print predictions and actuals:
            print(
                '> Human accuracy {:.5f} %'
                ''.format(general.binary_accuracy(
                    predictions, test_classes) * 100))

        # Return
        return {
            'loss': (accuracy * -1),
            'status': STATUS_OK,
            'estimated_accuracy': accuracy,
            'hyperparameters': params}

    def cleanup(self):
        """Release memory and delete checkpoint files.

        Args:
            None

        Returns:
            None

        """
        # Delete
        os.remove(self._path_checkpoint)

    def stationary(self):
        """Evaluate wether the timeseries is stationary.

        non-stationary timeseries are probably random walks and not
        suitable for forecasting.

        Args:
            None

        Returns:
            state: True if stationary

        """
        # Initialize key variables
        state = False
        values = []

        # statistical test
        result = adfuller(self._y_current)
        adf = result[0]
        print('> Stationarity Test:')
        print('  ADF Statistic: {:.3f}'.format(adf))
        print('  p-value: {:.3f}'.format(result[1]))
        print('  Critical Values:')
        for key, value in result[4].items():
            print('\t{}: {:.3f}'.format(key, value))
            values.append(value)

        # Return
        if adf < min(values):
            state = True
        print('  Stationarity: {}'.format(state))
        return state

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

    def plot_train(self, model, start_idx, length=100):
        """Plot the predicted and true output-signals.

        Args:
            model: Training model
            start_idx: Start-index for the time-series.
            length: Sequence-length to process and plot.

        Returns:
            None

        """
        # Plot
        self._plot_comparison(model, start_idx, length=length, train=True)

    def plot_test(self, model, start_idx, length=100):
        """Plot the predicted and true output-signals.

        Args:
            model: Training model
            start_idx: Start-index for the time-series.
            length: Sequence-length to process and plot.

        Returns:
            None

        """
        # Plot
        self._plot_comparison(model, start_idx, length=length, train=False)

    def _plot_comparison(self, model, start_idx, length=100, train=True):
        """Plot the predicted and true output-signals.

        Args:
            model: Training model
            start_idx: Start-index for the time-series.
            length: Sequence-length to process and plot.
            train: Boolean whether to use training- or test-set.

        Returns:
            None

        """
        # Initialize key variables
        datetimes = {}
        num_train = self.training_rows

        # Don't plot if we are looking at binary classes
        if bool(self._binary) is True:
            print('> Will not plot charts for binary class values.')
            return

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
            y_true = self._y_test[start_idx:end_idx]
            shim = 'Test'

            # Datetimes to use for testing
            datetimes[shim] = self._data.datetime()[
                -self.test_rows-1:][start_idx:end_idx]

        # Input-signals for the model.
        x_values = np.expand_dims(x_values, axis=0)

        # Use the model to predict the output-signals.
        y_pred = model.predict(x_values)

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
                    -self.test_rows:][start_idx:]

                # The number of datetimes for the 'actual' plot must match
                # that of current values
                datetimes['actual'] = self._data.datetime()[
                    -self.test_rows:][start_idx:]

            # Create a filename
            filename = (
                '/tmp/batch_{}_epochs_{}_training_{}_{}_{}_{}.png').format(
                    self.hyperparameters['batch_size'],
                    self.hyperparameters['epochs'],
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
            if self._display is True:
                fig.savefig(filename, bbox_inches='tight')
                plt.show()
            else:
                fig.savefig(filename, bbox_inches='tight')
            print('> Saving file: {}'.format(filename))

            # Close figure
            plt.close(fig=fig)


class ModelMGPU(Model):
    '''
    https://github.com/keras-team/keras/issues/2436#issuecomment-354882296
    '''
    def __init__(self, ser_model, **kwargs):
        pmodel = multi_gpu_model(ser_model, **kwargs)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
