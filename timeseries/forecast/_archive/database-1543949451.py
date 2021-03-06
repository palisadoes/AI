"""Library to process the ingest of data files."""

# Standard imports
import sys
import time
import csv
from datetime import datetime

# PIP imports
import pandas as pd
import numpy as np


class _DataFile(object):
    """Class ingests file data."""

    def __init__(self, filename):
        """Intialize the class.

        Args:
            filename: Name of file
            symbol: Symbol to update

        Returns:
            None

        """
        # Initialize key variables
        self._filename = filename

    def filedata(self, ts_start=None, rrd_step=300):
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
        with open(self._filename, 'r') as csvfile:
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
                # Replace the popped item with one at the beginning of
                # the series
                data_dict[int(ts_min - (count * rrd_step))] = 0
            else:
                break
            count += 1

        # Return
        print('Records ingested:', len(data_dict))
        return data_dict


class DataGRU(_DataFile):
    """Prepare data for use by GRU models."""

    def __init__(self, filename, shift_steps):
        """Intialize the class.

        Args:
            filename: Name of file

        Returns:
            None

        """
        # Setup inheritance
        _DataFile.__init__(self, filename)

        # Initialize key variables
        self._shift_steps = shift_steps
        self._label2predict = 'close'

        # Process data
        self._dataframe = self.__dataframe()
        (self._vectors, self._classes) = self._vector_targets()

    def __dataframe(self):
        """Create vectors from data.

        Args:
            shift_steps: List of steps

        Returns:
            result: dataframe for learning

        """
        # Initialize key varialbes
        output = {
            'weekday': [],
            'day': [],
            'year': [],
            'month': [],
            'hour': [],
            'minute': [],
            'second': [],
            'timestamp': [],
            'value': []}

        # Create result to return
        result = pd.DataFrame(columns=[
            'year', 'month', 'weekday', 'day', 'timestamp',
            'hour', 'minute', 'second', 'value']).astype('float16')

        # Get list of values
        for epoch, value in sorted(self.filedata().items()):
            output['value'].append(value)
            output['timestamp'].append(epoch)
            output['day'].append(
                int(datetime.fromtimestamp(epoch).strftime('%d')))
            output['weekday'].append(
                int(datetime.fromtimestamp(epoch).strftime('%w')))
            output['hour'].append(
                int(datetime.fromtimestamp(epoch).strftime('%H')))
            output['minute'].append(
                int(datetime.fromtimestamp(epoch).strftime('%M')))
            output['second'].append(
                int(datetime.fromtimestamp(epoch).strftime('%S')))
            output['month'].append(
                int(datetime.fromtimestamp(epoch).strftime('%m')))
            output['year'].append(
                int(datetime.fromtimestamp(epoch).strftime('%Y')))

        # Add current value columns
        result['value'] = pd.Series(output['value'])
        result['day'] = pd.Series(output['day'])
        result['weekday'] = pd.Series(output['weekday'])
        result['year'] = pd.Series(output['year'])
        result['hour'] = pd.Series(output['hour'])
        result['month'] = pd.Series(output['month'])
        result['minute'] = pd.Series(output['minute'])
        result['second'] = pd.Series(output['second'])
        result['timestamp'] = pd.Series(output['timestamp'])

        # Return
        return result

    def datetime(self):
        """Create a numpy array of datetimes.

        Args:
            None

        Returns:
            result: numpy array of timestamps

        """
        # Initialize key variables
        result = pd.to_datetime(pd.DataFrame({
            'year': self._dataframe['year'],
            'month': self._dataframe['month'],
            'day': self._dataframe['day'],
            'hour': self._dataframe['hour'],
            'minute': self._dataframe['minute']})).values

        # Return
        return result

    def labels(self):
        """Get class labels.

        Args:
            None

        Returns:
            result: list of labels

        """
        # Return
        result = self._shift_steps
        return result

    def values(self):
        """Get close values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self._dataframe['value']
        return result

    def vectors(self):
        """Create a numpy array of vectors.

        Args:
            None

        Returns:
            result: Tuple of numpy array of vectors. (train, test)

        """
        # Return
        result = (self._vectors['train'], self._vectors['all'])

        '''for item in self._vectors['train']:
            print(item)
        sys.exit(0)'''

        return result

    def classes(self):
        """Create a numpy array of classes.

        Args:
            None

        Returns:
            result: numpy array of classes

        """
        # Return
        result = (self._classes['train'], self._classes['all'])
        return result

    def _vector_targets(self):
        """Create vectors and targets from data.

        Args:
            shift_steps: List of steps

        Returns:
            result: dataframe for learning

        """
        # Initialize key variables
        targets = {}
        columns = []
        crop_by = max(self._shift_steps)
        label2predict = 'value'
        x_data = {'train': None, 'all': None}
        y_data = {'train': None, 'all': None}
        desired_columns = [
            'weekday', 'day', 'hour', 'minute', 'value']

        # Remove all undesirable columns from the dataframe
        pandas_df = self._dataframe
        imported_columns = list(pandas_df)
        for column in imported_columns:
            if column not in desired_columns:
                pandas_df = pandas_df.drop(column, axis=1)

        # Create shifted values for learning
        for step in self._shift_steps:
            '''
            Note the negative time-shift!

            We want the future state targets to line up with the timestamp of
            the last value of each sample set.
            '''
            targets[step] = pandas_df[label2predict].shift(-step)
            columns.append(step)

        # Get class values for each vector
        classes = pd.DataFrame(columns=columns)
        for step in self._shift_steps:
            # Shift each column by the value of its label
            classes[step] = pandas_df[label2predict].shift(-step)

        # Create class and vector dataframes with only non NaN values
        # (val_loss won't improve otherwise)
        y_data['train'] = classes.values[:-crop_by]
        y_data['all'] = classes.values[:]
        x_data['train'] = pandas_df.values[:-crop_by]
        x_data['all'] = pandas_df.values[:]

        # Return
        return(x_data, y_data)


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
