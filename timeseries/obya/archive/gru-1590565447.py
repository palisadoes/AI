"""Module forecast data using RNN AI using GRU feedback."""

# Standard imports
from __future__ import print_function
import os
import sys
from copy import deepcopy
from pprint import pprint
import gc
from collections import namedtuple

# PIP3 imports.
import yaml
import numpy as np
from sklearn.metrics import mean_absolute_error
from hyperopt import STATUS_OK

# TensorFlow imports
import tensorflow as tf

# Keras imports
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.optimizers import RMSprop
from keras.initializers import RandomUniform
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from tensorflow.keras.backend import square, mean

# Custom package imports
from obya.model import memory
from obya.model import files
from obya import HyperParameters, WARMUP_STEPS


class Model():
    """Process data for ingestion.

    Roughly based on:

    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb

    """

    def __init__(
            self, _data, identifier, batch_size=256, epochs=20,
            sequence_length=500, dropout=0.1, divider=1, test_size=0.2,
            layers=1, patience=5, units=128, multigpu=False):
        """Instantiate the class.

        Args:
            data: etl.Data object
            batch_size: Size of batch
            sequence_length: Length of vectors for for each target
            display: Show charts of results if True

        Returns:
            None

        """
        # Setup memory
        self._processors = memory.setup()

        # Initialize key variables
        self._identifier = identifier
        if multigpu is True:
            self._gpus = len(self._processors.gpus)
        else:
            self._gpus = 1
        _batch_size = int(batch_size * self._gpus)
        self._test_size = test_size

        # Get data
        self._data = _data

        # Set steps per epoch
        normal = self._data.split()
        (training_rows, _) = normal.x_train.shape
        steps_per_epoch = int((training_rows // _batch_size) / divider)

        # Set key file locations
        self._files = files.files(identifier)

        # Initialize parameters
        self._hyperparameters = HyperParameters(
            units=abs(units),
            dropout=abs(dropout),
            layers=int(abs(layers)),
            sequence_length=abs(sequence_length),
            patience=abs(patience),
            batch_size=_batch_size,
            epochs=abs(epochs),
            steps_per_epoch=steps_per_epoch
        )

        # Delete any stale checkpoint file
        if os.path.exists(self._files.checkpoint) is True:
            os.remove(self._files.checkpoint)

    def info(self):
        """Print out information on the dataset.

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        normal = self._data.split()
        scaled = self._data.scaled_split()

        # Print stuff
        print('\n> Numpy Data Type: {}'.format(
            type(normal.x_train)))

        print('> Numpy Data Shape: {}'.format(
            normal.x_train.shape))

        print('> Numpy Data Row[0]: {}'.format(
            normal.x_train[0]))

        print('> Numpy Data Row[Last]: {}'.format(
            normal.x_train[-1]))

        print('> Numpy Targets Type: {}'.format(
            type(normal.y_train)))

        print('> Numpy Vector Feature Type: {}'.format(
            type(normal.y_train[0][0])))

        print('> Numpy Targets Shape: {}'.format(
            normal.y_train.shape))

        print('> Number of Samples: {}'.format(
            len(normal.x_train) + len(normal.x_test)))

        print('> Number of Training Samples: {}'.format(
            normal.x_train.shape[0]))

        print('> Number of Training Classes: {}'.format(
            len(normal.y_train)))

        print('> Number of Test Samples: {}'.format(len(normal.x_test)))

        print('> Training Minimum Value:', np.min(normal.x_train))

        print('> Training Maximum Value:', np.max(normal.x_train))

        print('> Number X signals: {}'.format(normal.x_train[1]))

        print('> Number Y signals: {}'.format(normal.y_train[1]))

        # Print epoch related data
        print('> Epochs:', self._hyperparameters.epochs)
        print('> Batch Size:', self._hyperparameters.batch_size)

        # Display estimated memory footprint of training data.
        print('> Data size: {:.2f} Bytes'.format(
            normal.x_train.nbytes))

        print('> Scaled Training Minimum Value: {}'.format(
            np.min(scaled.x_train)))

        print('> Scaled Training Maximum Value: {}'.format(
            np.max(scaled.x_train)))

        '''
        The data-set has now been prepared as 2-dimensional numpy arrays. The
        training-data has many observations, consisting of
        'x_feature_count' input-signals and 'y_feature_count' output-signals.

        These are the array-shapes of the input and output data:
        '''

        print('> Scaled Training Data Shape: {}'.format(
            scaled.x_train.shape))
        print('> Scaled Training Targets Shape: {}'.format(
            scaled.y_train.shape))

    def train(self, params=None):
        """Train the Recurrent Neural Network.

        Args:
            None

        Returns:
            _model: RNN model

        """
        # Initialize key variables
        use_sigmoid = True

        # Intialize key variables realted to data
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

        # Prepare the generator
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
            np.expand_dims(scaled.x_test, axis=0),
            np.expand_dims(scaled.y_test, axis=0)
        )

        '''
        Instantiate the base model (or "template" model).
        We recommend doing this with under a CPU device scope,
        so that the model's weights are hosted on CPU memory.
        Otherwise they may end up hosted on a GPU, which would
        complicate weight sharing.

        NOTE: multi_gpu_model values will be way off if you don't do this.
        '''
        if bool(self._processors.gpus) is True:
            with tf.device(self._processors.gpus[0]):
                ai_model = Sequential()
        else:
            with tf.device(self._processors.cpus[0]):
                ai_model = Sequential()

        '''
        We can now add a Gated Recurrent Unit (GRU) to the network. This will
        have 'units' outputs for each time-step in the sequence.

        Note that because this is the first layer in the model, Keras needs to
        know the shape of its input, which is a batch of sequences of arbitrary
        length (indicated by None), where each observation has a number of
        input-signals (x_feature_count).
        '''

        ai_model.add(GRU(
            _hyperparameters.units,
            return_sequences=True,
            recurrent_dropout=_hyperparameters.dropout,
            input_shape=(None, x_feature_count)))

        for _ in range(1, abs(_hyperparameters.layers)):
            ai_model.add(GRU(
                _hyperparameters.units,
                recurrent_dropout=_hyperparameters.dropout,
                return_sequences=True))

        '''
        The GRU outputs a batch from keras_contrib.layers.advanced_activations
        of sequences of 'units' values. We want to predict
        'y_feature_count' output-signals, so we add a fully-connected (or
        dense) layer which maps 'units' values down to only 'y_feature_count'
        values.

        The output-signals in the data-set have been limited to be between 0
        and 1 using a scaler-object. So we also limit the output of the neural
        network using the Sigmoid activation function, which squashes the
        output to be between 0 and 1.
        '''

        if bool(use_sigmoid) is True:
            ai_model.add(Dense(y_feature_count, activation='sigmoid'))

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
            ai_model.add(Dense(
                y_feature_count,
                activation='linear',
                kernel_initializer=init))

        # Compile Model

        '''
        This is the optimizer and the beginning learning-rate that we will use.
        We then compile the Keras model so it is ready for training.
        '''

        optimizer = RMSprop(lr=1e-3)
        ai_model.compile(
            loss=model_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Display layers

        '''
        This is a very small model with only two layers. The output shape of
        (None, None, 'y_feature_count') means that the model will output a
        batch with an arbitrary number of sequences, each of which has an
        arbitrary number of observations, and each observation has
        'y_feature_count' signals. This corresponds to the 'y_feature_count'
        target signals we want to predict.
        '''
        print('\n> Summary (Parallel):\n')
        print(ai_model.summary())

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
            log_dir=self._files.log_dir, histogram_freq=0, write_graph=False
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

        history = ai_model.fit(
            x=generator,
            epochs=_hyperparameters.epochs,
            steps_per_epoch=_hyperparameters.steps_per_epoch,
            validation_data=validation_data,
            callbacks=callbacks)

        # Save model
        self.save(ai_model, history)

    def mtrain(self, params=None):
        """Train the Recurrent Neural Network.

        Args:
            None

        Returns:
            _model: RNN model

        """
        # Intialize key variables realted to data
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

        # Prepare the generator
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
            np.expand_dims(scaled.x_test, axis=0),
            np.expand_dims(scaled.y_test, axis=0)
        )

        # Get model
        ai_model = self._compile(_hyperparameters)

        # Callback Functions
        callbacks = _callbacks(self._files, _hyperparameters.patience)

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

        history = ai_model.fit(
            x=generator,
            epochs=_hyperparameters.epochs,
            steps_per_epoch=_hyperparameters.steps_per_epoch,
            validation_data=validation_data,
            callbacks=callbacks)

        # Save model
        self.save(ai_model, history)

    def _compile(self, _hyperparameters):
        """Compile the default model.

        Args:
            None

        Returns:
            ai_model: RNN model

        """
        # Initialize key variables
        use_sigmoid = False

        '''
        Instantiate the base model (or "template" model).
        We recommend doing this with under a CPU device scope,
        so that the model's weights are hosted on CPU memory.
        Otherwise they may end up hosted on a GPU, which would
        complicate weight sharing.
        '''

        # Intialize key variables realted to data
        normal = self._data.split()
        (_, x_feature_count) = normal.x_train.shape
        (_, y_feature_count) = normal.y_train.shape

        # Get GPU information
        devices = memory.setup()
        gpus = devices.gpus[:max(1, len(devices.gpus) - 1)]
        cpus = devices.cpus[0]

        # Start creating the model
        strategy = tf.distribute.MirroredStrategy(
            gpus,
            cross_device_ops=tf.distribute.ReductionToOneDevice(
                reduce_to_device=cpus))
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        # Define the model
        with strategy.scope():
            ai_model = tf.keras.Sequential()

            '''
            We can now add a Gated Recurrent Unit (GRU) to the network. This
            will have 'units' outputs for each time-step in the sequence.

            Note that because this is the first layer in the model, Keras
            needs to know the shape of its input, which is a batch of sequences
            of arbitrary length (indicated by None), where each observation
            has a number of input-signals (x_feature_count).
            '''

            ai_model.add(
                tf.keras.layers.GRU(
                    _hyperparameters.units,
                    return_sequences=True,
                    recurrent_dropout=_hyperparameters.dropout,
                    input_shape=(None, x_feature_count))
            )

            for _ in range(1, abs(_hyperparameters.layers)):
                ai_model.add(
                    tf.keras.layers.GRU(
                        _hyperparameters.units,
                        return_sequences=True,
                        recurrent_dropout=_hyperparameters.dropout)
                )

            '''
            The GRU outputs a batch from
            keras_contrib.layers.advanced_activations of sequences of 'units'
            values. We want to predict 'y_feature_count' output-signals, so we
            add a fully-connected (or dense) layer which maps 'units' values
            down to only 'y_feature_count' values.

            The output-signals in the data-set have been limited to be between
            0 and 1 using a scaler-object. So we also limit the output of the
            neural network using the Sigmoid activation function, which
            squashes the output to be between 0 and 1.
            '''

            if bool(use_sigmoid) is True:
                ai_model.add(
                    tf.keras.layers.Dense(
                        y_feature_count, activation='sigmoid'
                    )
                )

            '''
            A problem with using the Sigmoid activation function, is that we
            can now only output values in the same range as the training-data.

            For example, if the training-data only has values between -20 and
            +30, then the scaler-object will map -20 to 0 and +30 to 1. So if
            we limit the output of the neural network to be between 0 and 1
            using the Sigmoid function, this can only be mapped back to values
            between -20 and +30.

            We can use a linear activation function on the output instead. This
            allows for the output to take on arbitrary values. It might work
            with the standard initialization for a simple network architecture,
            but for more complicated network architectures e.g. with more
            layers, it might be necessary to initialize the weights with
            smaller values to avoid NaN values during training. You may need to
            experiment with this to get it working.
            '''

            if bool(use_sigmoid) is False:
                init = tf.keras.initializers.RandomUniform(
                    minval=-0.03, maxval=0.03)
                ai_model.add(tf.keras.layers.Dense(
                    y_feature_count,
                    activation='linear',
                    kernel_initializer=init))

            # Compile Model

            '''
            This is the optimizer and the beginning learning-rate that we will
            use. We then compile the Keras model so it is ready for training.
            '''

            optimizer = tf.keras.optimizers.RMSprop(lr=1e-3)
            ai_model.compile(
                loss=model_loss,
                optimizer=optimizer,
                metrics=['accuracy'])

            # Display layers

            '''
            This is a very small model with only two layers. The output shape
            of (None, None, 'y_feature_count') means that the model will output
            a batch with an arbitrary number of sequences, each of which has an
            arbitrary number of observations, and each observation has
            'y_feature_count' signals. This corresponds to the
            'y_feature_count' target signals we want to predict.
            '''
            print('\n> Summary (Parallel):\n')
            print(ai_model.summary())

        return ai_model

    def save(self, _model, history):
        """Save the Recurrent Neural Network model.

        Args:
            None

        Returns:
            _model: RNN model

        """
        # Save history to file
        with open(self._files.history, 'w') as yaml_file:
            _data = {}
            for key, value_list in history.history.items():
                _data[key] = np.array(value_list).tolist()
            yaml.dump(_data, yaml_file)

        # Serialize model to HDF5
        _model.save(self._files.model_parameters, save_format='tf')

        # Serialize weights to HDF5
        _model.save_weights(self._files.model_weights)
        print('> Saved model to disk')

    def evaluate(self):
        """Evaluate the model.

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        scaled = self._data.scaled_split()

        # Get model from disk
        _model = files.load_model(self._identifier)

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
            loss=model_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Performance on Test-Set

        '''
        We can now evaluate the model's performance on the validation-set.
        This function expects a batch of data, but we will just use one long
        time-series for the test-set, so we just expand the
        array-dimensionality to create a batch with that one sequence.
        '''

        x_scaled = scaled.x_test
        y_scaled = scaled.y_test

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
        scaled = self._data.scaled_split()
        scaled_vectors = scaled.x_test
        test_classes = scaled.y_test

        # Create model
        model = deepcopy(self.train(params=params))

        # Input-signals for the model.
        x_values = np.expand_dims(scaled_vectors, axis=0)

        # Get the predictions
        predictions_scaled = model.predict(x_values, verbose=1)

        # The output of the model is between 0 and 1.
        # Do an inverse map to get it back to the scale
        # of the original data-set.
        predictions = scaled.y_scaler.inverse_transform(predictions_scaled[0])

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
        if os.path.exists(self._files.checkpoint):
            os.remove(self._files.checkpoint)


def model_loss(y_true, y_pred):
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
    # The shape of both input tensors are:
    # [batch_size, sequence_length, y_feature_count].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, WARMUP_STEPS:, :]
    y_pred_slice = y_pred[:, WARMUP_STEPS:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, y_feature_count]

    # Calculate the Mean Squared Error and use it as loss.
    mse = mean(square(y_true_slice - y_pred_slice))
    return mse


def _batch_generator(parameters):
    """Create generator function to create random batches of training-data.

    Args:
        parameters: Named tuple of parameters to be used

    Returns:

        result: Tuple of (x_batch, y_batch) where:

            x_batch: Numpy array of 'batch_size' groups of 'sequence_length'
                vectors where each vector has 'x_feature_count' features

            y_batch: Numpy array of 'batch_size' groups of 'sequence_length'
                vectors where each vector has 'y_feature_count' features

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

        result = (x_batch, y_batch)
        yield result


def _callbacks(_files, patience):
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
        filepath=_files.checkpoint, monitor='val_loss',
        verbose=1, save_weights_only=True, save_best_only=True
    )

    '''
    This is the callback for stopping the optimization when performance
    worsens on the validation-set.
    '''

    callback_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, verbose=1
    )

    '''
    This is the callback for writing the TensorBoard log during training.
    '''

    callback_tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=_files.log_dir, histogram_freq=0, write_graph=False
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


def load(identifier):
    """Load the Recurrent Neural Network model from disk.

    Args:
        identifier: Identifier of model to load

    Returns:
        result: RNN model

    """
    # Turn off verbose logging
    memory.setup()

    # Setup a Load object
    Load = namedtuple(
        'Load', 'model, history')

    # Initialize key Variables
    _files = files.files(identifier)

    # Check for the existence of model files and directories
    if os.path.isdir(_files.model_parameters) is False:
        print('''\
Directory {} not found. Please create it by training your model first.\
'''.format(_files.model_parameters))
        sys.exit(0)

    if os.path.isfile(_files.model_weights) is False:
        print('File {} not found.'.format(_files.model_weights))
        sys.exit(0)

    # Load yaml and create model
    '''
    You have to use this custom_object parameter to read loss values from the
    customized loss function cannot be save to a keras model. It does not seem
    to work if the load function is in a module different from the one in which
    the customized loss function is located. Reference:

    https://github.com/keras-team/keras/issues/9377#issuecomment-396187881
    '''

    ai_model = tf.keras.models.load_model(
        _files.model_parameters,
        custom_objects={'model_loss': model_loss})

    # Load weights into new model
    ai_model.load_weights(_files.model_weights, by_name=True)

    # Load yaml and create model
    with open(_files.history, 'r') as yaml_file:
        history = yaml.safe_load(yaml_file)

    result = Load(model=ai_model, history=history)
    return result
