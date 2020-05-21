#!/usr/bin/env python3
"""Script to forecast data using RNN AI using GRU feedback."""

# Standard imports
from __future__ import print_function
import time
import os
from copy import deepcopy
from pprint import pprint
import gc
from collections import namedtuple

# PIP3 imports.
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from hyperopt import STATUS_OK

# TensorFlow imports
import tensorflow as tf

# Keras imports
from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, GRU
from keras.optimizers import RMSprop
from keras.initializers import RandomUniform
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.utils import multi_gpu_model
from tensorflow.keras.backend import square, mean

# Custom package imports
from obya.model import memory


class Model(object):
    """Process data for ingestion.

    Roughly based on:

    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb

    """

    def __init__(
            self, _data, batch_size=64, epochs=20,
            sequence_length=20, warmup_steps=50, dropout=0.2,
            layers=3, patience=5, units=256, display=False,
            multigpu=False):
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
        gpus = memory.setup()

        # Initialize key variables
        self._warmup_steps = warmup_steps
        self._display = display
        if multigpu is True:
            self._gpus = gpus
        else:
            self._gpus = 1

        # Set key file locations
        path_prefix = '/tmp/keras-{}'.format(int(time.time()))
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
            batch_size=int(batch_size * self._gpus),
            epochs=abs(epochs)
        )

        # Delete any stale checkpoint file
        if os.path.exists(self._files.checkpoint) is True:
            os.remove(self._files.checkpoint)

        # Get data
        self._split = _data.split()
        self._scaled_split = _data.scaled_split()

    def info(self):
        """Print out information on the dataset.

        Args:
            None

        Returns:
            None

        """
        # Print stuff
        print('\n> Numpy Data Type: {}'.format(
            type(self._split.x_train.values)))

        print('> Numpy Data Shape: {}'.format(
            self._split.x_train.values.shape))

        print('> Numpy Data Row[0]: {}'.format(
            self._split.x_train.values[0]))

        print('> Numpy Data Row[Last]: {}'.format(
            self._split.x_train.values[-1]))

        print('> Numpy Targets Type: {}'.format(
            type(self._split.y_train.values)))

        print('> Numpy Vector Feature Type: {}'.format(
            type(self._split.y_train.values[0][0])))

        print('> Numpy Targets Shape: {}'.format(
            self._split.y_train.values.shape))

        print('> Number of Samples: {}'.format(
            len(self._split.x_train) + len(
                self._split.x_test) + len(self._split.x_validate)))

        print('> Number of Training Samples: {}'.format(
            self._split.x_train.values.shape[0]))

        print('> Number of Training Classes: {}'.format(
            len(self._split.y_train)))

        print('> Number of Test Samples: {}'.format(len(self._split.x_test)))

        print('> Training Minimum Value:', np.min(self._split.x_train.values))

        print('> Training Maximum Value:', np.max(self._split.x_train.values))

        print('> Number X signals: {}'.format(self._split.x_train.values[1]))

        print('> Number Y signals: {}'.format(self._split.y_train.values[1]))

        # Print epoch related data
        print('> Epochs:', self._hyperparameters.epochs)
        print('> Batch Size:', self._hyperparameters.batch_size)

        # Display estimated memory footprint of training data.
        print('> Data size: {:.2f} Bytes'.format(
            self._split.x_train.values.nbytes))

        print('> Scaled Training Minimum Value: {}'.format(
            np.min(self._scaled_split.x_train)))

        print('> Scaled Training Maximum Value: {}'.format(
            np.max(self._scaled_split.x_train)))

        '''
        The data-set has now been prepared as 2-dimensional numpy arrays. The
        training-data has almost 300k observations, consisting of 20
        input-signals and 3 output-signals.

        These are the array-shapes of the input and output data:
        '''

        print('> Scaled Training Data Shape: {}'.format(
            self._scaled_split.x_train.shape))
        print('> Scaled Training Targets Shape: {}'.format(
            self._scaled_split.y_train.shape))

    def model(self, params=None):
        """Create the Recurrent Neural Network.

        Args:
            None

        Returns:
            _model: RNN model

        """
        # Initialize key variables
        use_sigmoid = False
        (training_rows,
         vector_count_features) = self._split.x_train.values.shape
        vector_count_classes = self._split.y_train.values.shape[1]

        if params is None:
            _hyperparameters = self._hyperparameters
        else:
            _hyperparameters = params
            _hyperparameters.batch_size = int(
                _hyperparameters.batch_size * self._gpus)

        # Calculate the steps per epoch
        epoch_steps = int(training_rows / _hyperparameters.batch_size) + 1

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

        # serial_model.add(GRU(
        #     _hyperparameters.units,
        #     return_sequences=True,
        #     recurrent_dropout=_hyperparameters.dropout,
        #     input_shape=(None, vector_count_features)))

        serial_model.add(GRU(
            _hyperparameters.units,
            return_sequences=True,
            recurrent_dropout=_hyperparameters.dropout,
            input_shape=(None, vector_count_features)))

        for _ in range(1, _hyperparameters.layers):
            serial_model.add(GRU(
                _hyperparameters.units,
                recurrent_dropout=_hyperparameters.dropout,
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

        if bool(use_sigmoid) is True:
            serial_model.add(
                Dense(vector_count_classes, activation='sigmoid'))

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

        if bool(use_sigmoid) is False:
            # Maybe use lower init-ranges.
            init = RandomUniform(minval=-0.05, maxval=0.05)

            serial_model.add(Dense(
                vector_count_classes,
                activation='linear',
                kernel_initializer=init))

        # Apply multi-GPU logic.
        if self._gpus == 1:
            parallel_model = serial_model
            print('> Training using single GPU.')
        else:
            try:
                # Use multiple GPUs
                parallel_model = multi_gpu_model(
                    serial_model,
                    cpu_relocation=True,
                    gpus=self._gpus)
                print('> Training using multiple GPUs.')
            except ValueError:
                parallel_model = serial_model
                print('> Single GPU detected. Training using single GPU.')

        # Compile Model

        '''
        This is the optimizer and the beginning learning-rate that we will use.
        We then compile the Keras model so it is ready for training.
        '''

        optimizer = RMSprop(lr=1e-3)
        parallel_model.compile(
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
        print('\n> Model Summary (Parallel):\n')
        print(parallel_model.summary())
        print('\n> Model Summary (Serial):\n')
        print(serial_model.summary())

        # Create the batch-generator.
        generator = self._batch_generator(
            _hyperparameters.batch_size,
            _hyperparameters.sequence_length)

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

        validation_data = (
            np.expand_dims(self._scaled_split.x_train, axis=0),
            np.expand_dims(self._scaled_split.y_train, axis=0)
        )

        # Callback Functions

        '''
        During training we want to save checkpoints and log the progress to
        TensorBoard so we create the appropriate callbacks for Keras.

        This is the callback for writing checkpoints during training.
        '''

        callback_checkpoint = ModelCheckpoint(
            filepath=self._files.checkpoint, monitor='val_loss',
            verbose=1, save_weights_only=True, save_best_only=True
        )

        '''
        This is the callback for stopping the optimization when performance
        worsens on the validation-set.
        '''

        callback_early_stopping = EarlyStopping(
            monitor='val_loss', patience=_hyperparameters.patience, verbose=1
        )

        '''
        This is the callback for writing the TensorBoard log during training.
        '''

        callback_tensorboard = TensorBoard(
            log_dir='/tmp/23_logs/', histogram_freq=0, write_graph=False
        )

        '''
        This callback reduces the learning-rate for the optimizer if the
        validation-loss has not improved since the last epoch
        (as indicated by patience=0). The learning-rate will be reduced by
        multiplying it with the given factor. We set a start learning-rate of
        1e-3 above, so multiplying it by 0.1 gives a learning-rate of 1e-4.
        We don't want the learning-rate to go any lower than this.
        '''

        callback_reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, min_lr=1e-4, patience=0, verbose=1
        )

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

        history = parallel_model.fit_generator(
            generator=generator,
            epochs=_hyperparameters.epochs,
            steps_per_epoch=epoch_steps,
            use_multiprocessing=True,
            validation_data=validation_data,
            callbacks=callbacks)

        print("Ploting History")
        plt.plot(history.history['loss'], label='Parallel Training Loss')
        plt.plot(history.history['val_loss'], label='Parallel Validation Loss')
        plt.legend()
        plt.show()

        # Return
        return parallel_model

    def save(self, _model):
        """Save the Recurrent Neural Network model.

        Args:
            None

        Returns:
            _model: RNN model

        """
        # Serialize model to JSON
        model_yaml = _model.to_yaml()
        with open(self._files.model_parameters, 'w') as yaml_file:
            yaml_file.write(model_yaml)

        # Serialize weights to HDF5
        _model.save_weights(self._files.model_weights)
        print('> Saved model to disk')

    def load_model(self):
        """Load the Recurrent Neural Network model from disk.

        Args:
            None

        Returns:
            _model: RNN model

        """
        # Load yaml and create model
        print('> Loading model from disk')
        with open(self._files.model_parameters, 'r') as yaml_file:
            loaded_model_yaml = yaml_file.read()
        _model = model_from_yaml(loaded_model_yaml)

        # Load weights into new model
        _model.load_weights(self._files.model_weights, by_name=True)
        print('> Finished loading model from disk')

        # Return
        return _model

    def evaluate(self, _model):
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

        if os.path.exists(self._files.checkpoint):
            _model.load_weights(self._files.checkpoint)

        optimizer = RMSprop(lr=1e-3)
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

        x_scaled = self._scaled_split.x_test.values
        y_scaled = self._scaled_split.y_test.values

        # Evaluate the MSE accuracy
        result = _model.evaluate(
            x=np.expand_dims(x_scaled, axis=0),
            y=np.expand_dims(y_scaled, axis=0))

        # If you have several metrics you can use this instead.
        print('> Metrics (test-set):')
        for _value, _metric in zip(result, _model.metrics_names):
            print('\t{}: {:.10f}'.format(_metric, _value))

    def objective(self, params=None):
        """Optimize the Recurrent Neural Network.

        Args:
            None

        Returns:
            _model: RNN model

        """
        # Initialize key variables
        model = deepcopy(self.model(params=params))
        scaled_vectors = self._scaled_split.x_test.values
        test_classes = self._split.y_test

        # Input-signals for the model.
        x_values = np.expand_dims(scaled_vectors, axis=0)

        # Get the predictions
        predictions_scaled = model.predict(x_values, verbose=1)

        # The output of the model is between 0 and 1.
        # Do an inverse map to get it back to the scale
        # of the original data-set.
        predictions = self._scaled_split.y_scaler.inverse_transform(
            predictions_scaled[0])

        # Get the error value
        accuracy = mean_absolute_error(test_classes, predictions)

        # Free object memory
        del model
        gc.collect()

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
        os.remove(self._files.checkpoint)

    def _batch_generator(self, batch_size, sequence_length):
        """Create generator function to create random batches of training-data.

        Args:
            batch_size: Size of batch
            sequence_length: Length of sequence

        Returns:
            (x_batch, y_batch)

        """
        # Intialize key variables
        (training_rows,
         vector_count_features) = self._split.x_train.values.shape
        vector_count_classes = self._split.y_train.values.shape[1]

        # Infinite loop.
        while True:
            # Allocate a new array for the batch of input-signals.
            # Number of features in x_train.
            x_shape = (
                batch_size, sequence_length, vector_count_features)
            x_batch = np.zeros(shape=x_shape, dtype=np.float16)

            # Allocate a new array for the batch of output-signals.
            # Number of features in y_train.
            y_shape = (
                batch_size, sequence_length, vector_count_classes)
            y_batch = np.zeros(shape=y_shape, dtype=np.float16)

            # Fill the batch with random sequences of data.
            for i in range(batch_size):
                # Get a random start-index.
                # This points somewhere into the training-data.
                idx = np.random.randint(training_rows - sequence_length)

                # Copy the sequences of data starting at this index.
                x_batch[i] = self._scaled_split.x_train[
                    idx:idx + sequence_length]
                y_batch[i] = self._scaled_split.y_train[
                    idx:idx + sequence_length]

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
            mse: Mean Squared Error

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

        # Calculate the Mean Squared Error and use it as loss.
        mse = mean(square(y_true_slice - y_pred_slice))

        # Return
        return mse
