#!/usr/bin/env python3
"""Script to demonstrate linear regression."""

# Standard importations
import math
import datetime
import pickle
import tempfile
import os

# PIP importations
import quandl
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

# Specify the style to use for matplotlib plots
style.use('ggplot')


def main():
    """Main Function.

    Display data prediction from linear regression model

    """
    # Initialize key variables
    test_size_percent = 20

    # Get data
    data = Data()
    x_ary = data.x
    y_ary = data.y
    x_lately = data.lately
    dataframe = data.dataframe

    # Create test and training data automatically
    (X_train, X_test, y_train, y_test) = model_selection.train_test_split(
        x_ary, y_ary, test_size=test_size_percent/100)

    # Create a linear classifier and add (fit) data to it for training
    _clf = LinearRegression()
    _clf.fit(X_train, y_train)

    # Pickle the data and retrieve (illustrative) using a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False)
    with open(temp.name,'wb') as pickle_out:
        pickle.dump(_clf, pickle_out)
    with open(temp.name,'rb') as pickle_in:
        clf = pickle.load(pickle_in)
    os.remove(temp.name)

    # Get the accuracy
    accuracy = clf.score(X_test, y_test)

    # Create an array of forecasted values
    forecast_set = clf.predict(x_lately)

    print(forecast_set, accuracy)

    # Plot the data
    plotdata(dataframe, forecast_set)


def plotdata(dataframe, forecast_set):
    """Function to plot the data we have.

    Args:
        dataframe: Dataframe with original data
        forecast_set: Set of forecasted data

    Returns:
        None

    """
    # Add a forecast column to the dataframe
    dataframe['Forecast'] = np.nan

    # Get the date based on the last date in the dataframe
    last_date = dataframe.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    # Append rows to the dataset with forecast data with the correct date and
    # make sure the non 'Forecast' rows are NaN
    for item in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        dataframe.loc[next_date] = [
            np.nan for _ in range(len(dataframe.columns) - 1)] + [item]

    # Plot data
    dataframe['Adj. Close'].plot()
    dataframe['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


class Data(object):
    """Class to get data."""

    def __init__(self):
        """Method to instantiate the class and get the data to analyze.

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        forecast_col = 'Adj. Close'
        forecast_pct = 0.01

        # Get data, Google daily stock price
        dataframe = quandl.get('WIKI/GOOGL')

        # Filter so we only get a few desired columns using the header row value
        dataframe = dataframe[
            ['Adj. Open', 'Adj. High', 'Adj. Low',
             'Adj. Close', 'Adj. Volume']]

        #  Add new column (High / Low percentage change over the day)
        dataframe['HL_PCT'] = (
            (dataframe['Adj. High'] - dataframe['Adj. Low']) / dataframe[
                'Adj. Close']) * 100.0

        #  Add new column (Daily percentage change)
        dataframe['PCT_change'] = ((
            dataframe['Adj. Close'] - dataframe['Adj. Open']) / dataframe[
                'Adj. Open']) * 100.0

        # Do some further filtering to only have these headings
        dataframe = dataframe[
            ['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

        # Fill any None or NaN values with an outlier value of -99999
        dataframe.fillna(value=-99999, inplace=True)

        # Determine how may days into the future we want to go.
        # In this case forecast_pct days ahead.
        forecast_out = int(math.ceil(forecast_pct * len(dataframe)))

        """
        Add a new column 'label' with the values in forecast_col, but
        forecast_out days into the future. In other words you are taking a
        current value but assigning it to a label in the past, thereby
        making it a prediction.
        """
        dataframe['label'] = dataframe[forecast_col].shift(-forecast_out)

        # Separate the features from the labels and convert to numpy arrays
        x_ary = np.array(dataframe.drop(['label'], 1))

        """
        Normalize the vectors from 0 to 1. This standardization of datasets is
        a common requirement for many machine learning estimators
        """
        x_ary = preprocessing.scale(x_ary)
        x_ary = x_ary[:-forecast_out]

        # We need the leftover X values from the shift to do accuracy testing
        x_lately = x_ary[-forecast_out:]

        """
        Drop any new NaN information because of the label shift above
        NOTE: We could have done this to get the same result:

        y_ary = np.array(dataframe['label'][:-forecast_out])
        """
        dataframe.dropna(inplace=True)
        y_ary = np.array(dataframe['label'])

        # Make variables global
        self.x = x_ary
        self.y = y_ary
        self.dataframe = dataframe
        self.lately = x_lately


if __name__ == "__main__":
    main()
