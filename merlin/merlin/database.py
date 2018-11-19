"""Library to process the ingest of data files."""

import sys
import os
import time
import datetime
import zipfile
import shutil

# PIP imports
import pandas as pd

# Append custom application libraries
from merlin import log


class ReadFile(object):
    """Class ingests file data."""

    def __init__(self, filename):
        """Function for intializing the class.

        Args:
            filename: Name of file
            symbol: Symbol to update

        Returns:
            None

        """
        # Initialize key variables
        self.filename = filename
        self.symbol = ''

    def valid(self):
        """Process file data.

        Args:
            None

        Returns:
            validity: True if basic file validity checks pass.

        """
        # Initialize key variables
        validity = True
        return validity

    def data(self):
        """Process file data.

        Args:
            None

        Returns:
            result: dataframe for learning

        """
        # Read data
        headings = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        data = pd.read_csv(self.filename, names=headings)
        data = data.drop(['time'], axis=1)

        # Get date values from data
        weekday = pd.to_datetime(data['date'], format='%Y.%m.%d').dt.weekday
        day = pd.to_datetime(data['date'], format='%Y.%m.%d').dt.day
        dayofyear = pd.to_datetime(
            data['date'], format='%Y.%m.%d').dt.dayofyear
        quarter = pd.to_datetime(data['date'], format='%Y.%m.%d').dt.quarter
        month = pd.to_datetime(data['date'], format='%Y.%m.%d').dt.month

        # Calculate the percentage and real differences between columns
        num_difference = data.drop(['date'], axis=1).diff()
        pct_difference = data.drop(['date'], axis=1).pct_change()

        # Create result to return
        result = pd.DataFrame(columns=[
            'open', 'high', 'low', 'close',
            'weekday', 'day', 'dayofyear', 'quarter', 'month', 'num_diff_open',
            'num_diff_high', 'num_diff_low', 'num_diff_close', 'pct_diff_open',
            'pct_diff_high', 'pct_diff_low', 'pct_diff_close'])
        result['open'] = data['open']
        result['high'] = data['high']
        result['low'] = data['low']
        result['close'] = data['close']
        result['day'] = day
        result['weekday'] = weekday
        result['dayofyear'] = dayofyear
        result['quarter'] = quarter
        result['month'] = month
        result['weekday'] = weekday
        result['day'] = day
        result['dayofyear'] = dayofyear
        result['quarter'] = quarter
        result['month'] = month
        result['num_diff_open'] = num_difference['open']
        result['num_diff_high'] = num_difference['high']
        result['num_diff_low'] = num_difference['low']
        result['num_diff_close'] = num_difference['close']
        result['pct_diff_open'] = pct_difference['open']
        result['pct_diff_high'] = pct_difference['high']
        result['pct_diff_low'] = pct_difference['low']
        result['pct_diff_close'] = pct_difference['close']

        # Delete the first row of the dataframe as it has NaN values from the
        # .diff() and .pct_change() operations
        result = result.iloc[1:]

        # Return
        return result

    def vector_targets(self, shift_steps=[1]):
        """Create vectors and targets from data.

        Args:
            None

        Returns:
            result: dataframe for learning

        """
        # Initialize key variables
        pandas_df = self.data()
        targets = {}
        columns = []
        label2predict = 'close'

        # Create column labels for dataframe columns
        # Create shifted values for learning
        for step in shift_steps:
            '''
            Note the negative time-shift!

            We want the future state targets to line up with the timestamp of
            the last value of each sample set.
            '''
            targets[step] = pandas_df[label2predict].shift(-step)
            columns.append('{}'.format(step))

        # Get vectors
        x_data = pandas_df.values[0:-max(shift_steps)]

        # Get class values for each vector
        classes = pd.DataFrame(columns=columns)
        for step in shift_steps:
            # Shift each column by the value of its label
            classes[str(step)] = pandas_df[label2predict].shift(-step)
        # Create dataframe with only non NaN values
        y_data = classes.values[:-max(shift_steps)]

        # Return
        return(x_data, y_data)
