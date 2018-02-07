#!/usr/bin/env python3
"""Script to forecast data using CNN AI."""

# Standard imports
import sys
import argparse
import csv
from pprint import pprint
import time

# Pip imports
import numpy as np
import pandas
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


class _Data(object):
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
        self._values = []
        self._lag = lag

        # Get list of values
        for _, value in sorted(data.items()):
            self._values.append(value)

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
        columns = [dataframe.shift(index) for index in range(1, self._lag + 1)]
        columns.append(dataframe)
        dataframe = pandas.concat(columns, axis=1)
        dataframe.fillna(0, inplace=True)
        return dataframe.values

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
        values = self._values

        # Create the difference and return
        diff = []
        for index in range(interval, len(values)):
            value = values[index] - values[index - interval]
            diff.append(value)
        return pandas.Series(diff).values

    def load(self):
        """Load data for RNN.

        Args:
            None

        Returns:
            (train, test): Tuple of list of trainint and test data

        """
        # Initialize key variables
        data = self._timeseries_to_supervised()
        index = round(0.9 * len(data))

        # Return
        train = data[0:-index]
        test = data[-index:]
        return (train, test)

    def scale(self):
        # Initialize key variables
        (_train, _test) = self.load()

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


class Data(object):
    """Process data for ingestion."""

    def __init__(self, data, features=30, forecast=1, base=5):
        """Method that instantiates the class.

        Args:
            data: Dict of values keyed by timestamp
            features: Number of features per vector
            forecast: Forecast horizon
            base: Round up to the base int(X)

        Returns:
            None

        """
        # Initialize key variables
        self._features = features
        self._forecast = forecast
        self._base = base
        self._values = []

        # Get list of values
        for _, value in sorted(data.items()):
            self._values.append(value)

    def load(self):
        """Load RNN data.

            Based on tutorial at:
            http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        sequence_length = self._features + 1
        result = []

        # Create an numpy array of values rounded to the nearest
        # self._base value
        for index in range(len(self._values) - sequence_length):
            result.append(self._values[index: index + sequence_length])
        result = np.array(result)
        for _ in np.nditer(result, op_flags=['readwrite']):
            _[...] = roundup(_, self._base)

        # Use the first 90% of result values as training data
        row = round(0.9 * result.shape[0])
        train = result[:int(row), :]
        np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]

        # Use the last 10% of result values as test data
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1]

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Return
        return (x_train, y_train, x_test, y_test)


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
    return maxes


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


def build_model(features, feature_dimensions, hidden_layer_neurons):
    """Build an LSTM model.

    Args:
        features: Number of features per vector
        feature_dimensions: Dimension of features (Number of rows per feature)
        hidden_layer_neurons: Number of neurons per layers

    Returns:
        model: Graph of LSTM model

    """
    # Initialize key variables
    start = time.time()

    # Let's do some learning!
    model = Sequential()

    '''
    The Long Short-Term Memory network (LSTM) is a type of Recurrent Neural
    Network (RNN).

    A benefit of this type of network is that it can learn and remember over
    long sequences and does not rely on a pre-specified window lagged
    observation as input.

    In Keras, this is referred to as stateful, and involves setting the
    "stateful" argument to "True" when defining an LSTM layer.

    By default, an LSTM layer in Keras maintains state between data within
    one batch. A batch of data is a fixed-sized number of rows from the
    training dataset that defines how many patterns to process before updating
    the weights of the network. State in the LSTM layer between batches is
    cleared by default, therefore we must make the LSTM stateful. This gives
    us fine-grained control over when state of the LSTM layer is cleared,
    by calling the reset_states() function.
    '''
    model.add(
        LSTM(
            units=hidden_layer_neurons,
            input_shape=(features, feature_dimensions),
            return_sequences=True,
            stateful=True
        )
    )
    model.add(Dropout(0.2))

    model.add(
        LSTM(
            units=hidden_layer_neurons,
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
    model.add(Activation('linear'))

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
    print('> Compilation Time:', time.time() - start)
    return model


def main():
    """Main Function.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    periods = 288           # No. of periods per vector for predictions
    features = 7            # No. of features per vector
    forecast = 1            # No. of periods into the future for predictions
    base = 2                # Round up to the base int(X)
    rrd_step = 300
    forecast = 5
    feature_dimensions = 1
    ts_start = int(time.time())

    '''
    The batch size is often much smaller than the total number of samples.
    It, along with the number of epochs, defines how quickly the network learns
    the data (how often the weights are updated).
    '''
    batch_size = 128
    epochs = 5

    '''
    The final import parameter in defining the LSTM layer is the number of
    neurons, also called the number of memory units or blocks.
    '''
    hidden_layer_neurons = 512

    # Get filename
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='Name of CSV file to read.',
        type=str, required=True)
    args = parser.parse_args()
    filename = args.filename

    file_data = read_file(filename, rrd_step=rrd_step)
    data = _Data(file_data)
    (scaler, train, test) = data.scale()
    print(train)
    sys.exit(0)

    ######################################################################

    # Open file and get data
    print('> Loading data... ')
    file_data = read_file(filename, rrd_step=rrd_step)
    maxes_data = get_maxes(file_data, periods=periods)
    data = Data(maxes_data, forecast=forecast, base=base, features=features)
    (x_train, y_train, x_test, y_test) = data.load()

    # Create the data model
    print('> Data Loaded. Compiling...', x_train.shape, y_train.shape)
    model = build_model(features, feature_dimensions, hidden_layer_neurons)

    '''
    Once compiled, the network can be fit to the training data. Because the
    network is stateful, we must control when the internal state is reset.
    Therefore, we must manually manage the training process one epoch at a
    time across the desired number of epochs.

    By default, the samples within an epoch are shuffled prior to being exposed
    to the network. Again, this is undesirable for the LSTM because we want the
    network to build up state as it learns across the sequence of observations.
    We can disable the shuffling of samples by setting "shuffle" to "False".
    '''
    for _ in range(epochs):
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            shuffle=False,
            epochs=1,
            validation_split=0.05)
        model.reset_states()

    print('->', x_train.shape, x_test.shape)
    #score, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    predicted = model.predict(x_test)
    predicted = np.reshape(predicted, (predicted.size,))

    matching = 0
    for index, prediction in enumerate(predicted):
        value = roundup(prediction, base)
        expected = roundup(y_test[index], base)
        print('Prediction: {}\tActual: {}'.format(value, expected))
        if value == expected:
            matching += 1

    '''matching = 0
    for index, vector in enumerate(x_test):
        _value = model.predict(vector[np.newaxis, :, :])[0, 0]
        value = roundup(_value, base)
        expected = roundup(y_test[index], base)
        print(index, len(x_test), expected, value, _value)
        #print(type(expected), type(value))
        #sys.exit(0)
        if value == expected:
            matching += 1'''

    print('\nAccuracy: \t\t{}%'.format(round(100 * matching/len(x_test), 3)))
    print('Duration: \t\t{}s'.format(int(time.time()) - ts_start))
    print('Days per Vector: \t\t{}'.format(features))

    #print('Score: ', score)


if __name__ == "__main__":
    main()
