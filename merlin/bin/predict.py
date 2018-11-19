#!/usr/bin/env python3
"""Script to forecast data using RNN AI using GRU feedback."""

# Standard imports
import argparse
import csv
import sys
from datetime import datetime
import time
import traceback

# PIP3 imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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


class RNNGRU(object):
    """Process data for ingestion."""

    def __init__(
            self, data, periods=288, batch_size=64, sequence_length=20,
            warmup_steps=50, epochs=20, display=False):
        """Instantiate the class.

        Args:
            data: Tuple of (x_data, y_data)
            periods: Number of timestamp data points per vector
            batch_size: Size of batch
            sequence_length: Length of vectors for for each target
            warmup_steps:

        Returns:
            None

        """
        # Initialize key variables
        self.periods = periods
        self.target_names = ['one', 'two', 'three', 'four', 'five']
        self.target_names = ['one']
        self.warmup_steps = warmup_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.display = display

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
        (x_data, y_data) = data

        print('\n> Numpy Data Type: {}'.format(type(x_data)))
        print("> Numpy Data Shape: {}".format(x_data.shape))
        print("> Numpy Data Row[0]: {}".format(x_data[0]))
        print('> Numpy Targets Type: {}'.format(type(y_data)))
        print("> Numpy Targets Shape: {}".format(y_data.shape))

        '''
        This is the number of observations (aka. data-points or samples) in
        the data-set:
        '''

        num_data = len(x_data)

        '''
        This is the fraction of the data-set that will be used for the
        training-set:
        '''

        train_split = 0.9

        '''
        This is the number of observations in the training-set:
        '''

        self.num_train = int(train_split * num_data)

        '''
        This is the number of observations in the test-set:
        '''

        num_test = num_data - self.num_train

        print('> Number of Samples: {}'.format(num_data))
        print("> Number of Training Samples: {}".format(self.num_train))
        print("> Number of Test Samples: {}".format(num_test))

        # Create test and training data
        x_train = x_data[0:self.num_train]
        x_test = x_data[self.num_train:]
        self.y_train = y_data[0:self.num_train]
        self.y_test = y_data[self.num_train:]
        self.num_x_signals = x_data.shape[1]
        self.num_y_signals = y_data.shape[1]

        print("> Training Minimum Value:", np.min(x_train))
        print("> Training Maximum Value:", np.max(x_train))

        '''
        steps_per_epoch is the number of batch iterations before a training
        epoch is considered finished.
        '''

        self.steps_per_epoch = int(self.num_train / batch_size) + 1
        print("> Epochs:", epochs)
        print("> Batch Size:", batch_size)
        print("> Steps:", self.steps_per_epoch)

        '''
        Calculate the estimated memory footprint.
        '''

        print("> Data size: {:.2f} Bytes".format(x_data.nbytes))

        '''
        The neural network works best on values roughly between -1 and 1, so we
        need to scale the data before it is being input to the neural network.
        We can use scikit-learn for this.

        We first create a scaler-object for the input-signals.

        Then we detect the range of values from the training-data and scale
        the training-data.
        '''

        x_scaler = MinMaxScaler()
        self.x_train_scaled = x_scaler.fit_transform(x_train)

        print('> Scaled Training Minimum Value: {}'.format(
            np.min(self.x_train_scaled)))
        print('> Scaled Training Maximum Value: {}'.format(
            np.max(self.x_train_scaled)))

        self.x_test_scaled = x_scaler.transform(x_test)

        '''
        The target-data comes from the same data-set as the input-signals,
        because it is the weather-data for one of the cities that is merely
        time-shifted. But the target-data could be from a different source with
        different value-ranges, so we create a separate scaler-object for the
        target-data.
        '''

        self.y_scaler = MinMaxScaler()
        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train)
        y_test_scaled = self.y_scaler.transform(self.y_test)

        # Data Generator

        '''
        The data-set has now been prepared as 2-dimensional numpy arrays. The
        training-data has almost 300k observations, consisting of 20
        input-signals and 3 output-signals.

        These are the array-shapes of the input and output data:
        '''

        print('> Scaled Training Data Shape: {}'.format(
            self.x_train_scaled.shape))
        print('> Scaled Training Targets Shape: {}'.format(
            self.y_train_scaled.shape))

        # We then create the batch-generator.

        generator = self.batch_generator(batch_size, sequence_length)

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

        validation_data = (np.expand_dims(self.x_test_scaled, axis=0),
                           np.expand_dims(y_test_scaled, axis=0))

        # Create the Recurrent Neural Network

        self.model = Sequential()

        '''
        We can now add a Gated Recurrent Unit (GRU) to the network. This will
        have 512 outputs for each time-step in the sequence.

        Note that because this is the first layer in the model, Keras needs to
        know the shape of its input, which is a batch of sequences of arbitrary
        length (indicated by None), where each observation has a number of
        input-signals (num_x_signals).
        '''

        self.model.add(GRU(
            units=1024,
            return_sequences=True,
            input_shape=(None, self.num_x_signals,)))

        '''
        The GRU outputs a batch of sequences of 512 values. We want to predict
        3 output-signals, so we add a fully-connected (or dense) layer which
        maps 512 values down to only 3 values.

        The output-signals in the data-set have been limited to be between 0
        and 1 using a scaler-object. So we also limit the output of the neural
        network using the Sigmoid activation function, which squashes the
        output to be between 0 and 1.'''

        self.model.add(Dense(self.num_y_signals, activation='sigmoid'))

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

            self.model.add(Dense(
                self.num_y_signals,
                activation='linear',
                kernel_initializer=init))

        # Compile Model

        '''
        This is the optimizer and the beginning learning-rate that we will use.
        We then compile the Keras model so it is ready for training.
        '''
        optimizer = RMSprop(lr=1e-3)
        self.model.compile(loss=self.loss_mse_warmup, optimizer=optimizer)

        '''
        This is a very small model with only two layers. The output shape of
        (None, None, 3) means that the model will output a batch with an
        arbitrary number of sequences, each of which has an arbitrary number of
        observations, and each observation has 3 signals. This corresponds to
        the 3 target signals we want to predict.
        '''
        print('> Model Summary:\n')
        print(self.model.summary())

        # Callback Functions

        '''
        During training we want to save checkpoints and log the progress to
        TensorBoard so we create the appropriate callbacks for Keras.

        This is the callback for writing checkpoints during training.
        '''

        path_checkpoint = '/tmp/23_checkpoint.keras'
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
                                                patience=5, verbose=1)

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

        try:
            self.model.fit_generator(
                generator=generator,
                epochs=self.epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_data=validation_data,
                callbacks=callbacks)
        except Exception as error:
            print('\n>{}\n'.format(error))
            traceback.print_exc()
            sys.exit(0)

        # Load Checkpoint

        '''
        Because we use early-stopping when training the model, it is possible
        that the model's performance has worsened on the test-set for several
        epochs before training was stopped. We therefore reload the last saved
        checkpoint, which should have the best performance on the test-set.
        '''

        print('> Loading model weights')

        try:
            self.model.load_weights(path_checkpoint)
        except Exception as error:
            print('\n> Error trying to load checkpoint.\n\n{}'.format(error))
            traceback.print_exc()
            sys.exit(0)

        # Performance on Test-Set

        '''
        We can now evaluate the model's performance on the test-set. This
        function expects a batch of data, but we will just use one long
        time-series for the test-set, so we just expand the
        array-dimensionality to create a batch with that one sequence.
        '''

        result = self.model.evaluate(
            x=np.expand_dims(self.x_test_scaled, axis=0),
            y=np.expand_dims(y_test_scaled, axis=0))

        print('> Loss (test-set): {}'.format(result))

        # If you have several metrics you can use this instead.
        if False:
            for res, metric in zip(result, self.model.metrics_names):
                print('{0}: {1:.3e}'.format(metric, res))

    def batch_generator(self, batch_size, sequence_length):
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
            x_shape = (batch_size, sequence_length, self.num_x_signals)
            x_batch = np.zeros(shape=x_shape, dtype=np.float16)

            # Allocate a new array for the batch of output-signals.
            y_shape = (batch_size, sequence_length, self.num_y_signals)
            y_batch = np.zeros(shape=y_shape, dtype=np.float16)

            # Fill the batch with random sequences of data.
            for i in range(batch_size):
                # Get a random start-index.
                # This points somewhere into the training-data.
                idx = np.random.randint(self.num_train - sequence_length)

                # Copy the sequences of data starting at this index.
                x_batch[i] = self.x_train_scaled[idx:idx+sequence_length]
                y_batch[i] = self.y_train_scaled[idx:idx+sequence_length]

            yield (x_batch, y_batch)

    def loss_mse_warmup(self, y_true, y_pred):
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
        warmup_steps = self.warmup_steps

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

    def plot_comparison(self, start_idx, length=100, train=True):
        """Plot the predicted and true output-signals.

        Args:
            start_idx: Start-index for the time-series.
            length: Sequence-length to process and plot.
            train: Boolean whether to use training- or test-set.

        Returns:
            None

        """
        if train:
            # Use training-data.
            x_values = self.x_train_scaled
            y_true = self.y_train
            shim = 'Train'
        else:
            # Use test-data.
            x_values = self.x_test_scaled
            y_true = self.y_test
            shim = 'Test'

        # End-index for the sequences.
        end_idx = start_idx + length

        # Select the sequences from the given start-index and
        # of the given length.
        x_values = x_values[start_idx:end_idx]
        y_true = y_true[start_idx:end_idx]

        # Input-signals for the model.
        x_values = np.expand_dims(x_values, axis=0)

        # Use the model to predict the output-signals.
        y_pred = self.model.predict(x_values)

        # The output of the model is between 0 and 1.
        # Do an inverse map to get it back to the scale
        # of the original data-set.
        y_pred_rescaled = self.y_scaler.inverse_transform(y_pred[0])

        # For each output-signal.
        for signal in range(len(self.target_names)):
            # Create a filename
            filename = (
                '/tmp/batch_{}_epochs_{}_training_{}_{}_{}_{}.png').format(
                    self.batch_size, self.epochs, self.num_train, signal,
                    int(time.time()), shim)

            # Get the output-signal predicted by the model.
            signal_pred = y_pred_rescaled[:, signal]

            # Get the true output-signal from the data-set.
            signal_true = y_true[:, signal]

            # Make the plotting-canvas bigger.
            plt.figure(figsize=(15, 5))

            # Plot and compare the two signals.
            plt.plot(signal_true, label='true')
            plt.plot(signal_pred, label='pred')

            # Plot grey box for warmup-period.
            _ = plt.axvspan(
                0, self.warmup_steps, facecolor='black', alpha=0.15)

            # Plot labels etc.
            plt.ylabel(self.target_names[signal])
            plt.legend()

            # Show and save the image
            if self.display is True:
                plt.savefig(filename, bbox_inches='tight')
                plt.show()
            else:
                plt.savefig(filename, bbox_inches='tight')
            print('> Saving file: {}'.format(filename))


def convert_data(data, periods, target_names):
    """Get data to analyze.

    Args:
        data: Dict of values keyed by epoch timestamp
        periods: Number of periods to shift data to get forecasts for Y values
        target_names: Name of dataframe column to use for Y values

    Returns:
        (x_data, y_data): X and Y values as numpy arrays

    """
    # Initialize key variables
    shift_steps = periods
    output = {
        'dow': [],
        'dom': [],
        'hour': [],
        'minute': [],
        'second': [],
        'value': []}

    # Get list of values
    for epoch, value in sorted(data.items()):
        output['value'].append(value)
        '''output['doy'].append(
            int(datetime.fromtimestamp(epoch).strftime('%j')))'''
        output['dom'].append(
            int(datetime.fromtimestamp(epoch).strftime('%d')))
        output['dow'].append(
            int(datetime.fromtimestamp(epoch).strftime('%w')))
        output['hour'].append(
            int(datetime.fromtimestamp(epoch).strftime('%H')))
        output['minute'].append(
            int(datetime.fromtimestamp(epoch).strftime('%M')))
        output['second'].append(
            int(datetime.fromtimestamp(epoch).strftime('%S')))

    # Convert to dataframe
    pandas_df = pd.DataFrame.from_dict(output)

    '''
    Note the negative time-shift!

    We want the future state targets to line up with the timestamp of the
    last value of each sample set.
    '''

    df_targets = pandas_df[target_names].shift(-shift_steps)
    x_data = pandas_df.values[0:-shift_steps]
    y_data = df_targets.values[:-shift_steps]

    # Return
    return(x_data, y_data)


def read_file(filename, ts_start=None, rrd_step=300):
    """Read data from file.

    Args:
        filename: Name of CSV file to read
        ts_start: Starting timestamp for which data should be retrieved
        rrd_step: Default RRD step time of the CSV file

    Returns:
        data_dict: Dict of values keyed by timestamp

    """
    # Initialize key variables
    data_dict = {}
    now = _normalize(int(time.time()), rrd_step)
    count = 1

    # Set the start time to be 2 years by default
    if (ts_start is None) or (ts_start < 0):
        ts_start = now - (3600 * 24 * 365 * 2)
    else:
        ts_start = _normalize(ts_start, rrd_step)

    # Read data
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            timestamp = _normalize(int(row[0]), rrd_step)
            if ts_start <= timestamp:
                value = float(row[1])
                data_dict[timestamp] = value

    # Fill in the blanks in timestamps
    ts_max = max(data_dict.keys())
    ts_min = min(data_dict.keys())
    timestamps = range(
        _normalize(ts_min, rrd_step),
        _normalize(ts_max, rrd_step),
        rrd_step)
    for timestamp in timestamps:
        if timestamp not in data_dict:
            data_dict[timestamp] = 0

    # Back track from the end of the data and delete any zero values until
    # no zeros are found. Sometimes the most recent csv data is zero due to
    # update delays. Replace the deleted entries with a zero value at the
    # beginning of the series
    _timestamps = sorted(data_dict.keys(), reverse=True)
    for timestamp in _timestamps:
        if bool(data_dict[timestamp]) is False:
            data_dict.pop(timestamp, None)
            # Replace the popped item with one at the beginning of the series
            data_dict[int(ts_min - (count * rrd_step))] = 0
        else:
            break
        count += 1

    # Return
    print('Records ingested:', len(data_dict))
    return data_dict


def _normalize(timestamp, rrd_step=300):
    """Normalize the timestamp to nearest rrd_step value.

    Args:
        rrd_step: RRD tool step value

    Returns:
        result: Normalized timestamp

    """
    # Return
    result = int((timestamp // rrd_step) * rrd_step)
    return result


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
        '-b', '--batch-size', help='Size of batch.',
        type=int, default=500)
    parser.add_argument(
        '-w', '--weeks',
        help='Number of weeks of data to consider when learning.',
        type=int, default=53)
    parser.add_argument(
        '-e', '--epochs',
        help='Number of epoch iterations to use.',
        type=int, default=20)
    parser.add_argument(
        '--display',
        help='Display on screen if True',
        action='store_true')
    args = parser.parse_args()
    filename = args.filename
    display = args.display

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
    sequence_length = 7 * periods * weeks

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
    _db = database.ReadFile(filename)
    data = _db.vector_targets([1, 2, 3, 4, 5])
    data = _db.vector_targets([1])

    # Do training
    rnn = RNNGRU(
        data,
        periods=periods,
        batch_size=batch_size,
        sequence_length=sequence_length,
        epochs=epochs,
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

    rnn.plot_comparison(start_idx=1, length=1000, train=True)

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

    rnn.plot_comparison(start_idx=1, length=200, train=False)

    # Print duration
    print("> Duration: {}s".format(duration))


if __name__ == "__main__":
    main()
