#!/usr/bin/env python3
"""Script to forecast timeseries data."""

# Standard imports
from __future__ import print_function
import os
import argparse
import sys

# Non-standard imports from ubuntu packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Non-standard imports using pip
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf


def main():
    """Process data.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    history = 50
    future = 10
    df_columns = ['close']

    # Setup parameters
    setup()

    # Import data
    args = arguments()
    df_ = pd.read_csv(args.filename)

    # Preprocessing data
    df_ = df_.set_index('date')[df_columns].tail(1000)
    df_ = df_.set_index(pd.to_datetime(df_.index))

    model = Model(df_, history=history, future=future)
    model.train()
    model.plot_predicted_vs_actual(-1)

class Model():
    """Preprocess the data for models."""

    def __init__(
            self, df_, batch_size=32, history=50, future=10, units=30,
            epochs=50, dropout_percent=20, model_file='/tmp/models-x.h5',
            validation_split=0.3):
        """Create feature array using historical data.

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        self._dropout_rate = dropout_percent / 100
        self._validation_split = validation_split
        self._df_ = df_
        self._batch_size = batch_size
        self._history = history
        self._future = future
        self._units = units
        self._epochs = epochs
        self._model_file = model_file
        self._input_features = len(self._df_.columns)

        # Preprocess the data
        (self.x_train, self.y_train,
         self.scaler) = self._data()

    def create(self):
        """Create Model.

        Args:
            None

        Returns:
            model: LSTM model

        """
        # Create model
        model = Sequential()
        model.add(LSTM(
            units=self._units,
            return_sequences=True,
            input_shape=(self._history, self._input_features)))
        model.add(Dropout(self._dropout_rate))
        model.add(LSTM(units=self._units, return_sequences=True))
        model.add(Dropout(self._dropout_rate))
        model.add(LSTM(units=self._units))
        model.add(Dropout(self._dropout_rate))
        model.add(Dense(self._future))
        model.summary()
        model.compile(
            loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        return model

    def train(self):
        """Train Model.

        Args:
            None

        Returns:
            model: Fitted model

        """
        # Train and save the model
        if os.path.exists(self._model_file) is False:
            model = self.create()
            results = model.fit(
                self.x_train,
                self.y_train,
                epochs=self._epochs,
                batch_size=self._batch_size,
                validation_split=self._validation_split,
                shuffle=False)
            visualize_results(results)
            model.save(self._model_file)
        else:
            model = self.load()

    def load(self):
        """Load model.

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        model = None

        # Load the model
        if os.path.exists(self._model_file) is True:
            model = load_model(self._model_file)
        return model

    def plot_predicted_vs_actual(self, offset):
        """Plot predicted versus actual values for same timeframe.

        Args:
            offset: Offset in the training data

        Returns:
            None

        """
        # Create the feature on which to base the prediction
        vector = self.x_train[offset].reshape(
            1, self._history, self._input_features)

        # Get the prediction model
        model = self.load()

        # Create values for plots
        raw_prediction = model.predict(vector).tolist()[0]
        prediction = self.scaler.inverse_transform(
            np.array(raw_prediction).reshape(-1, 1)).tolist()
        actual = self.scaler.inverse_transform(
            self.y_train[offset].reshape(-1, 1)).tolist()

        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(prediction, label='Predicted')
        plt.plot(actual, label='Actual')
        plt.title('Predicted vs. Actual')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_forecast(self):
        """Plot forecasted values after end of training set.

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        p_df_end = self._df_.tail(self._history)
        a_df_end = self._df_.tail(self._future)
        periods = 10

        # Get the prediction model
        model = self.load()

        # Create the feature on which to base the prediction
        p_vectors = np.array(df_end).reshape(
            1, self._history, self._input_features)
        raw_predictions = model.predict(p_vectors).tolist()[0]
        predictions = self.scaler.inverse_transform(
            np.array(raw_predictions).reshape(-1, 1)).tolist()

        # Convert predictions to DataFrame for ease of plotting
        df_predictions = pd.DataFrame(
            predictions,
            index=pd.date_range(
                start=self._df_[-1],
                periods=len(predictions),
                freq='D'),
            columns=df_end.columns)

        # Get actual values
        raw_actuals = self.scaler.inverse_transform(np.array(a_df_end))
        df_actuals = pd.DataFrame(
            raw_actuals,
            index=(a_df_end.index),
            columns=a_df_end.columns
        ).append(df_predictions.head(1))

        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(df_predictions, label='Predicted')
        plt.plot(df_actuals, label='Actual')
        plt.title('Forecasting Next {} days'.format(self._future))
        plt.ylabel('Price')
        plt.ylabel('Date')
        plt.legend()
        plt.show()

    def _data(self):
        """Preprocess the data.

        Args:
            None

        Returns:
            None

        """
        # Create the test and training datasets formatted for keras
        process = PreProcessing(
            self._df_,
            future=self._future,
            history=self._history)
        result = process.data()
        return result


class PreProcessing():
    """Preprocess the data for models."""

    def __init__(self, df_, history=50, future=50):
        """Create feature array using historical data.

        Args:
            df_: Dataframe
            history: Number of features per row of the array. This is equal to
                the number of historical entries in each row.
            future: Number of future entries to predict.
            features: Number of features in Dataframe.

        Returns:
            None

        """
        # Reshape single column array to a list of lists for tensorflow
        self._df_ = df_
        self._history = history
        self._future = future

    def data(self):
        """Create feature array using historical data.

        Args:
            None

        Returns:
            result: feature array and scaler

        """
        # Create a scaler
        scaler = MinMaxScaler()
        features = len(self._df_.columns)
        df_ = pd.DataFrame(
            scaler.fit_transform(self._df_),
            columns=self._df_.columns,
            index=self._df_.index)

        # Create training data features
        (x_train, y_train) = create_features(
            list(df_.close), history=self._history, future=self._future)

        '''
        Reshape data for LSTM to have three dimensions namely:

            (rows processed during training,
             training features,
             columns of data extracted from the dataset)

        The resulting array makes each row of featues have a row of only a
        single entry (input feature from database)
        '''
        # return
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], features)
        result = (x_train, y_train, scaler)
        return result


def create_features(df_, history=50, future=50):
    """Create feature array using historical data.

    Args:
        df_: Dataframe
        history: Number of features per row of the array. This is equal to
            the number of historical entries in each row.
        future: Number of future entries to predict.

    Returns:
        result: feature array

    """
    # Initialize key variables
    historical = []
    forecast = []

    # Process data
    for index in range(len(df_)):
        end_history = index + history
        end_future = end_history + future

        if end_future > len(df_):
            break

        historical.append(df_[index: end_history])
        forecast.append(df_[end_history: end_future])

    # Return
    result = (np.array(historical), np.array(forecast))
    return result


def visualize_results(results):
    """Plot results.

    Args:
        results: Model results

    Returns:
        None

    """
    # Initialize key variables
    history = results.history

    # Plot Loss
    plt.figure(figsize=(8, 4))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


def setup():
    """Setup TensorFlow 2 operating parameters.

    Args:
        None

    Returns:
        None

    """
    # Initialize key variables
    memory_limit = 1024

    # Reduce error logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Limit Tensorflow v2 Limit GPU Memory usage
    # https://www.tensorflow.org/guide/gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if bool(gpus) is True:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for index, _ in enumerate(gpus):
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[index],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print('Physical GPUs: {}, Logical GPUs: {}'.format(
                len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


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


if __name__ == "__main__":
    main()
