#!/usr/bin/env python3
"""Script to forecast data using CNN AI."""

# Standard imports
import sys
import argparse
import csv
import os
from pprint import pprint
from math import sqrt
import time

# Pip imports
import numpy as np
import pandas
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class Data(object):
    """Process data for ingestion.

    Based on: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

    """

    def __init__(self, data, lag=1):
        """Method that instantiates the class.

        Args:
            data: Dict of values keyed by timestamp
            lag: Lag

        Returns:
            None

        """
        # Initialize key variables
        self.values = []
        self.timestamps = []
        self._lag = lag

        # Get list of values
        for timestamp, value in sorted(data.items()):
            self.values.append(value)
            self.timestamps.append(timestamp)

        # Get data and scaled data
        self.scaler, self.scaled_train, self.scaled_test = self._scaled_data()
        self.scaled = np.vstack((self.scaled_train, self.scaled_test))

    def _timeseries_to_supervised(self):
        """Transform Time Series to Supervised Learning.

        The LSTM model in Keras assumes that your data is divided into input
        (X) and output (y) components.

        For a time series problem, we can achieve this by using the observation
        from the last time step (t-1) as the input and the observation at the
        current time step (t) as the output.

        We can achieve this using the shift() function in Pandas that will
        push all values in a series down by a specified number places. We
        require a shift of 1 place, which will become the input variables. The
        time series as it stands will be the output variables.

        We can then concatenate these two series together to create a DataFrame
        ready for supervised learning. The pushed-down series will have a new
        position at the top with no value. A NaN (not a number) value will be
        used in this position. We will replace these NaN values with 0 values,
        which the LSTM model will have to learn as "the start of the series"
        or "I have no data here," as a month with zero sales on this dataset
        has not been observed.

        We take a NumPy array of the raw time series data and a lag or number
        of shifted series to create and use as inputs.

        Args:
            None

        Returns:
            dataframe: DataFrame

        """
        # Get difference values
        diff_values = self._difference()

        # Create series for supervised learning
        dataframe = pandas.DataFrame(diff_values)
        columns = [dataframe.shift(i) for i in range(1, self._lag + 1)]
        columns.append(dataframe)
        dataframe = pandas.concat(columns, axis=1)
        dataframe.fillna(0, inplace=True)
        return dataframe

    def _difference(self, interval=1):
        """Create a differenced series.

        The dataset may not be stationary. This means that there is a structure
        in the data that is dependent on the time. Specifically, there could
        be an increasing trend in the data.

        Stationary data is easier to model and will very likely result in more
        skillful forecasts.

        The trend can be removed from the observations, then added back to
        forecasts later to return the prediction to the original scale and
        calculate a comparable error score.

        A standard way to remove a trend is by differencing the data. That is
        the observation from the previous time step (t-1) is subtracted from
        the current observation (t). This removes the trend and we are left
        with a difference series, or the changes to the observations from one
        time step to the next.

        We can achieve this automatically using the diff() function in pandas.
        Alternatively, we can get finer grained control and write our own
        function to do this, which is preferred for its flexibility in
        this case.

        Note that the first observation in the series is skipped as there is no
        prior observation with which to calculate a differenced value.

        Args:
            None

        Returns:
            dataframe: DataFrame

        """
        # Initialize key variables
        values = self.values

        # Create the difference and return
        diff = []
        for index in range(interval, len(values)):
            value = values[index] - values[index - interval]
            diff.append(value)
        return pandas.Series(diff)

    def invert_difference(self, inverted_class, interval=1):
        """Invert difference data.

        Args:
            inverted_class: Class value whose scale has already been inverted.
            interval:

        Returns:
            value: Inverted data

        """
        value = inverted_class + self.values[-interval]
        return value

    def _data(self):
        """Load data.

        Args:
            None

        Returns:
            (train, test): Tuple of list of train and test data

        """
        # Initialize key variables
        _data = self._timeseries_to_supervised()
        data = _data.values
        index = round(0.9 * len(data))

        # Return
        train = data[:index]
        test = data[index:]
        return (train, test)

    def _scaled_data(self):
        """Load scaled data.

        Args:
            None

        Returns:
            (scaler, train, test): Tuple of list of train and test data

        """
        # Initialize key variables
        (_train, _test) = self._data()

        # Fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(_train)

        # Transform train
        train = _train.reshape(_train.shape[0], _train.shape[1])
        train_scaled = scaler.transform(train)

        # Transform test
        test = _test.reshape(_test.shape[0], _test.shape[1])
        test_scaled = scaler.transform(test)

        # Return
        return scaler, train_scaled, test_scaled

    def invert_scale(self, x_value, predicted):
        """Inverse scaling for a forecasted value.

        Args:
            x_value: X value
            predicted: Forecasted value

        Returns:
            result: Inverted value

        """
        new_row = [_ for _ in x_value] + [predicted]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = self.scaler.inverse_transform(array)
        result = inverted[0, -1]
        return result


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


def roundup(value, base):
    """Round value to nearest base value.

    Args:
        value: Value to round up
        base: Base to use

    Returns:
        result

    """
    # Initialize key variables
    _base = int(base)

    # Return
    _result = int(base * round(float(value)/base))
    result = np.float32(_result)
    return result


def get_maxes(data, periods=288):
    """Get maximum values from data.

    Args:
        data: Dict of values keyed by timestamp
        periods: Sliding window number periods over
            which maxes will be calculated

    Returns:
        maxes: Dict of values keyed by timestamp

    """
    # Initialize key variables
    maxes = {}
    values = []
    timestamps = []

    # Create easier to use lists of data to work with
    for timestamp, value in sorted(data.items()):
        values.append(value)
        timestamps.append(timestamp)

    # Find the maxes in value every X periods
    item_count = len(values)
    for index in range(0, item_count - periods + 1):
        endpointer = index + periods
        sample = values[index:endpointer]
        max_sample = max(sample)
        _timestamp = timestamps[endpointer - 1]
        maxes[_timestamp] = max_sample

    # Return
    print('Maxes created:', len(maxes))
    return maxes


def lstm_model(
        data, hidden_layer_neurons, epochs, batch_size=1,
        feature_dimensions=1, verbose=False):
    """Build an LSTM model.

    Args:
        data: Data frame of X, Y values
        hidden_layer_neurons: Number of neurons per layers
        epochs: Number of iterations for learning
        batch_size
        feature_dimensions: Dimension of features (Number of rows per feature)

    Returns:
        model: Graph of LSTM model

    """
    # Initialize key variables
    start = time.time()

    # Process the data for fitting
    x_values, y_values = data[:, 0: -1], data[:, -1]
    x_shaped = x_values.reshape(x_values.shape[0], 1, x_values.shape[1])

    # Let's do some learning!
    model = Sequential()

    '''
    The Long Short-Term Memory network (LSTM) is a type of Recurrent Neural
    Network (RNN).

    A benefit of this type of network is that it can learn and remember over
    long sequences and does not rely on a pre-specified window lagged
    observation as input.

    In Keras, this is referred to as being "stateful", and involves setting the
    "stateful" argument to "True" when defining an LSTM layer.

    By default, an LSTM layer in Keras maintains state between data within
    one batch. A batch of data is a fixed-sized number of rows from the
    training dataset that defines how many patterns (sequences) to process
    before updating the weights of the network.

    A state is:
        Where am I now inside a sequence? Which time step is it? How is this
        particular sequence behaving since its beginning up to now?

    A weight is: What do I know about the general behavior of all sequences
        I've seen so far?

    State in the LSTM layer between batches is cleared by default. This is
    undesirable therefore we must make the LSTM stateful. This gives us
    fine-grained control over when state of the LSTM layer is cleared, by
    calling the reset_states() function during the model.fit() method.

    LSTM networks can be stacked in Keras in the same way that other layer
    types can be stacked. One addition to the configuration that is required
    is that an LSTM layer prior to each subsequent LSTM layer must return the
    sequence. This can be done by setting the return_sequences parameter on
    the layer to True.

    batch_size denotes the subset size of your training sample (e.g. 100 out
    of 1000) which is going to be used in order to train the network during its
    learning process. Each batch trains network in a successive order, taking
    into account the updated weights coming from the appliance of the previous
    batch.

    return_sequence indicates if a recurrent layer of the network should return
    its entire output sequence (i.e. a sequence of vectors of specific
    dimension) to the next layer of the network, or just its last only output
    which is a single vector of the same dimension. This value can be useful
    for networks conforming with an RNN architecture.

    batch_input_shape defines that the sequential classification of the
    neural network can accept input data of the defined only batch size,
    restricting in that way the creation of any variable dimension vector.
    It is widely used in stacked LSTM networks. It is a tuple of (batch_size,
    timesteps, data_dimension)
    '''
    timesteps = x_shaped.shape[1]
    data_dimension = x_shaped.shape[2]

    # Add layers to the model
    model.add(
        LSTM(
            units=hidden_layer_neurons,
            batch_input_shape=(batch_size, timesteps, data_dimension),
            return_sequences=True,
            stateful=True
        )
    )
    model.add(Dropout(0.2))

    model.add(
        LSTM(
            units=hidden_layer_neurons,
            batch_input_shape=(batch_size, timesteps, data_dimension),
            return_sequences=False,
            stateful=True
        )
    )
    model.add(Dropout(0.2))

    model.add(
        Dense(
            units=feature_dimensions
        )
    )
    # model.add(Activation('linear'))

    '''
    Once the network is specified, it must be compiled into an efficient
    symbolic representation using a backend mathematical library,
    such as TensorFlow.

    In compiling the network, we must specify a loss function and optimization
    algorithm. We will use "mean_squared_error" or "mse" as the loss function
    as it closely matches RMSE that we will are interested in, and the
    efficient ADAM optimization algorithm.
    '''
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    '''
    Once the model is compiled, the network can be fit to the training data.
    Because the network is stateful, we must control when the internal state
    is reset. Therefore, we must manually manage the training process one epoch
    at a time across the desired number of epochs.

    By default, the samples within an epoch are shuffled prior to being exposed
    to the network. Again, this is undesirable for the LSTM because we want the
    network to build up state as it learns across the sequence of observations.
    We can disable the shuffling of samples by setting "shuffle" to "False".
    '''
    for _ in range(epochs):
        model.fit(
            x_shaped,
            y_values,
            batch_size=batch_size,
            shuffle=False,
            epochs=1,
            verbose=verbose,
            validation_split=0.05)

        '''
        When the fit process reaches the total length of the samples,
        model.reset_states() is called to reset the internal state at the end
        of the training epoch, ready for the next training iteration.

        This iteration will start training from the beginning of the dataset
        therefore state will need to be reset as the previous state would only
        be relevant to the prior epoch iteration.
        '''
        model.reset_states()

    print('\n> Training Time: {:20.2f}'.format(time.time() - start))
    return model


def forecast_lstm(model, feature_vector, batch_size=1):
    """Make a one-step forecast.

    Args:
        model: LSTM model
        X:
        batch_size: Size of batch

    Returns:
        result: Forecast

    """
    reshaped_vector = feature_vector.reshape(1, 1, len(feature_vector))
    y_value = model.predict(reshaped_vector, batch_size=batch_size)
    result = y_value[0, 0]
    return result

class ForecastLSTM(object):
    """Process data for ingestion.

    Based on: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

    """

    def __init__(self, model, data, scaled_data):
        """Method that instantiates the class.

        Args:
            model: Trained LSTM model
            data: Data object
            scaled_data: Scaled data array to be used for forecasting

        Returns:
            None

        """
        # Initialize key variables
        self._batch_size = 1
        self._model = model
        self._data = data

    def _forecast_lstm(self):
        """Make a one-step forecast.

        Args:
            model: LSTM model
            X:
            batch_size: Size of batch

        Returns:
            result: Forecast

        """
        # Initialize key variables
        model = self._model
        batch_size = self._batch_size

        # Process
        reshaped_vector = feature_vector.reshape(1, 1, len(feature_vector))
        y_value = model.predict(reshaped_vector, batch_size=batch_size)
        result = y_value[0, 0]
        return result

def main():
    """Main Function.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    base = 2                # Round up to the base int(X)
    rrd_step = 300
    verbose = True
    periods = 288
    ts_start = int(time.time())
    predictions = []

    # Set logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    '''
    The batch size is often much smaller than the total number of samples.
    It, along with the number of epochs, defines how quickly the network learns
    the data (how often the weights are updated).

    In a stateful network, you should only pass inputs with a number of samples
    that can be divided by the batch size. Hence we use "1".
    '''
    batch_size = 1
    epochs = 1

    '''
    The final import parameter in defining the LSTM layer is the number of
    neurons, also called the number of memory units or blocks.
    '''
    hidden_layer_neurons = 5

    # Get filename
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='Name of CSV file to read.',
        type=str, required=True)
    args = parser.parse_args()
    filename = args.filename

    # Read data from file
    print('\nProcessing file: {}'.format(os.path.basename(filename)))
    file_data = read_file(filename, rrd_step=rrd_step)
    file_data = get_maxes(file_data, periods=periods)

    # Prepare data for modeling
    data = Data(file_data)

    # Fit the data to the model
    model = lstm_model(
        data.scaled_train, hidden_layer_neurons, epochs, verbose=verbose)

    '''
    During training, the internal state is reset after each epoch.

    While forecasting, we will not want to reset the internal state between
    forecasts. In fact, we would like the model to build up state as we
    forecast each time step in the test dataset.

    This raises the question as to what would be a good initial state for the
    network prior to forecasting the test dataset. We will seed the state by
    making a prediction on all samples in the training dataset. In theory, the
    internal state should now be set up ready to forecast the next time step.

    In other words, the current state of the model (created during model.fit)
    isn't suitable for forecasting.

    We now forecast the entire training dataset to build up state for
    forecasting.
    '''
    print('Creating model state for forecasting.')
    train_reshaped = data.scaled_train[:, 0].reshape(
        len(data.scaled_train), 1, 1)
    model.predict(train_reshaped, batch_size=batch_size)
    print('Completed model state for forecasting.')

    # Walk-forward validation on the test data
    test_scaled = data.scaled_test
    for index_test in range(len(test_scaled)):
        # Make one-step forecast
        feature_vector = test_scaled[index_test, 0:-1]
        feature_class = test_scaled[index_test, -1]

        predicted_class_scaled = forecast_lstm(
            model, feature_vector, batch_size=batch_size)

        # Invert scaling
        predicted_class_differenced = data.invert_scale(
            feature_vector, predicted_class_scaled)

        # Invert differencing
        pointer_differenced = len(test_scaled) + 1 - index_test
        print(len(test_scaled), index_test, pointer_differenced)
        predicted_class = data.invert_difference(
            predicted_class_differenced, pointer_differenced)

        # Store forecast
        index = len(data.scaled_train) + index_test + 1
        predictions.append(predicted_class)
        expected = data.values[index]

        # Print status
        '''print(
            'Timestamp={}, Predicted={}, Expected={}'
            ''.format(data.timestamps[index], predicted_class, expected))'''

    # Calculate RMSE
    test_values = data.values[-len(test_scaled):]
    rmse = sqrt(mean_squared_error(test_values, predictions))
    print('Test RMSE: {:5.3f}'.format(rmse))

    # Calculate % accuacy
    found = []
    for idx, value in enumerate(test_values):
        found.append(int(round(value, 0) == round(predictions[idx], 0)))
    print('Accuracy: {:5.1f}%'.format((100 * sum(found)) / len(test_values)))

    # Print execution time
    print('Execution time: {:5.1f}s\n'.format(int(time.time() - ts_start)))

    # All done
    sys.exit(0)


if __name__ == "__main__":
    main()
