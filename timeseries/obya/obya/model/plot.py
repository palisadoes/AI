"""Module forecast data using RNN AI using GRU feedback."""

import time

# PIP3 imports.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Custom package imports
from obya.model import files
from obya import WARMUP_STEPS


class Plot():
    """Plot learned data.

    Roughly based on:

    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb

    """

    def __init__(self, _data, identifier):
        """Instantiate the class.

        Args:
            data: etl.Data object
            identifier: Unique identifier for the data

        Returns:
            None

        """
        # Set key file locations
        self._identifier = identifier
        self._time = int(time.time())
        self._files = files.files(identifier)

        # Get data
        self._data = _data

    def history(self):
        """Plot the predicted and true output-signals.

        Args:
            model: Training model
            start_idx: Start-index for the time-series.
            length: Sequence-length to process and plot.

        Returns:
            None

        """
        # Initialize key variables
        _history = files.load_history(self._identifier)

        # Plot
        plt.plot(_history['loss'], label='Parallel Training Loss')
        plt.plot(_history['val_loss'], label='Parallel Validation Loss')
        plt.legend()
        plt.show()

    def train(self, start_idx, length=100):
        """Plot the predicted and true output-signals.

        Args:
            start_idx: Start-index for the time-series.
            length: Sequence-length to process and plot.

        Returns:
            None

        """
        # Plot
        self._plot_comparison(start_idx, length=length, train=True)

    def test(self, start_idx, length=100):
        """Plot the predicted and true output-signals.

        Args:
            model: Training model
            start_idx: Start-index for the time-series.
            length: Sequence-length to process and plot.

        Returns:
            None

        """
        # Intialize key variables realted to data
        normal = self._data.split()
        (test_rows, _) = normal.x_test.shape

        # Plot
        self._plot_comparison(
            start_idx,
            length=min(test_rows, length),
            train=False)

    def _plot_comparison(self, start_idx, length=100, train=True):
        """Plot the predicted and true output-signals.

        Args:
            start_idx: Start-index for the time-series.
            length: Sequence-length to process and plot.
            train: Boolean whether to use training- or test-set.

        Returns:
            None

        """
        # Get model
        model = files.load_model(self._identifier)

        # Intialize key variables realted to data
        normal = self._data.split()
        scaled = self._data.scaled_split()
        y_combined = self._data.values()
        datetimes = self._data.datetimes()
        (training_rows, _) = normal.x_train.shape

        # End-index for the sequences.
        end_idx = start_idx + length

        # Variables for date formatting
        days = mdates.DayLocator()   # Every day
        months = mdates.MonthLocator()  # Every month
        months_format = mdates.DateFormatter('%b %Y %H:%M')
        days_format = mdates.DateFormatter('%d')

        # Assign other variables dependent on the type of data we are plotting
        if train is True:
            # Use training-data.
            x_values = scaled.x_train[start_idx:end_idx]
            y_true = normal.y_train[start_idx:end_idx]
            shim = 'Train'

            # Datetimes to use for training
            datetimes = datetimes[:training_rows][start_idx:end_idx]

            # Only get current values that are a part of the training data
            current = y_combined[:training_rows][start_idx:end_idx]

        else:
            # Use test-data.
            x_values = scaled.x_test[start_idx:end_idx]
            y_true = normal.y_test[start_idx:end_idx]
            shim = 'Test'

            # Datetimes to use for testing
            datetimes = datetimes[training_rows:][start_idx:end_idx]

            # Only get current values that are a part of the test data.
            current = y_combined[training_rows:][start_idx:end_idx]

        # Input-signals for the model.
        x_values = np.expand_dims(x_values, axis=0)

        # Use the model to predict the output-signals.
        y_pred = model.predict(x_values)

        # The output of the model is between 0 and 1.
        # Do an inverse map to get it back to the scale
        # of the original data-set.
        y_pred_rescaled = scaled.y_scaler.inverse_transform(y_pred[0])

        # For each output-signal.
        for signal in range(len(self._data.labels())):
            # Create a filename
            filename = self._filename(shim, signal)

            # Get the output-signal predicted by the model.
            signal_pred = y_pred_rescaled[:, signal]

            # Get the true output-signal from the data-set.
            signal_true = y_true[:, signal]

            # Create a new chart
            (fig, axis) = plt.subplots(figsize=(15, 5))

            # Plot and compare the two signals.
            axis.plot(
                datetimes[:len(signal_true)],
                signal_true,
                label='Current + {}'.format(self._data.labels()[signal]))
            axis.plot(
                datetimes[:len(signal_pred)],
                signal_pred,
                label='Prediction')
            axis.plot(datetimes, current, label='Current')

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
            if 0 <= start_idx < WARMUP_STEPS:
                plt.axvspan(
                    datetimes[start_idx],
                    datetimes[min(length, WARMUP_STEPS)],
                    facecolor='black', alpha=0.15)

            # Show and save the image
            fig.savefig(filename, bbox_inches='tight')
            plt.show()
            print('> Saving file: {}'.format(filename))

            # Close figure
            plt.close(fig=fig)

    def train_test(self, model):
        """Plot the predicted and true output-signals.

        Args:
            model: Training model
            start_idx: Start-index for the time-series.
            length: Sequence-length to process and plot.

        Returns:
            None

        """
        # Intialize key variables realted to data
        normal = self._data.split()
        scaled = self._data.scaled_split()

        # Initialize other key variables
        shim = 'Comparison'

        # Use test-data.
        x_values = scaled.x_test[:]
        y_true = normal.y_test[:]

        # Input-signals for the model.
        x_values = np.expand_dims(x_values, axis=0)

        # Use the model to predict the output-signals.
        y_pred = model.predict(x_values)

        # The output of the model is between 0 and 1.
        # Do an inverse map to get it back to the scale
        # of the original data-set.
        y_pred_rescaled = scaled.y_scaler.inverse_transform(y_pred[0])

        # For each output-signal.
        for signal in range(len(self._data.labels())):
            # Create a filename
            filename = self._filename(shim, signal)

            # Get the output-signal predicted by the model.
            signal_pred = y_pred_rescaled[:, signal]

            # Get the true output-signal from the data-set.
            signal_true = y_true[:, signal]

            # Create a new chart
            (fig, axis) = plt.subplots(figsize=(15, 5))

            # Plot and compare the two signals.
            plt.scatter(
                signal_pred[:len(signal_true)],
                signal_true,
                alpha=0.1,
                label=(
                    'Predicted vs. Actual +{}'.format(
                        self._data.labels()[signal])))

            # Set plot labels and titles
            axis.set_title(
                'Predicted vs. Actual ({0} Future Intervals)'.format(
                    self._data.labels()[signal]))
            axis.set_ylabel('Predicted')
            axis.set_xlabel('Actual')
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

            # Remove tick marks
            ax.tick_params(axis='both', which='both', length=0)

            # Show and save the image
            fig.savefig(filename, bbox_inches='tight')
            plt.show()
            print('> Saving file: {}'.format(filename))

            # Close figure
            plt.close(fig=fig)

    def _filename(self, title, label):
        """Plot the predicted and true output-signals.

        Args:
            title: Chart title
            label: Class being charted

        Returns:
            filename: Filename

        """
        # Return
        filename = ('''\
/tmp/obya_chart_({0})_({1})_{2}.png''').format(title, label, self._time)
        return filename
