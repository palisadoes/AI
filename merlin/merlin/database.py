"""Library to process the ingest of data files."""

# Standard imports
import sys

# PIP imports
import pandas as pd

# Append custom application libraries
from merlin import general
from merlin import math


class DataSource(object):
    """Super class handling data retrieval."""

    def __init__(self):
        """Intialize the class.

        Args:
            None

        Returns:
            None

        """
        # Setup classwide variables
        self._kwindow = 35
        self._dwindow = 5
        self._rsiwindow = self._kwindow
        self._ignore_row_count = max(1, self._kwindow + self._dwindow)

    def values(self):
        """Process file data.

        Args:
            None

        Returns:
            result: DataFrame of values

        """
        # Placeholder
        pass

    def dates(self):
        """Process file data.

        Args:
            None

        Returns:
            result: DataFrame of dates

        """
        # Placeholder
        pass

    def open(self):
        """Get open values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self.values()['open'][self._ignore_row_count:]
        return result

    def high(self):
        """Get high values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self.values()['high'][self._ignore_row_count:]
        return result

    def low(self):
        """Get low values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self.values()['low'][self._ignore_row_count:]
        return result

    def close(self):
        """Get close values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self.values()['close'][self._ignore_row_count:]
        return result

    def volume(self):
        """Get volume values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self.values()['volume'][self._ignore_row_count:]
        return result

    def dataframe(self):
        """Create vectors from data.

        Args:
            shift_steps: List of steps

        Returns:
            result: dataframe for learning

        """
        # Calculate the percentage and real differences between columns
        difference = math.Difference(self.values())
        num_difference = difference.actual()
        pct_difference = difference.relative()

        # Create result to return
        result = pd.DataFrame(columns=[
            'open', 'high', 'low', 'close',
            'weekday', 'day', 'dayofyear', 'quarter', 'month', 'num_diff_open',
            'num_diff_high', 'num_diff_low', 'num_diff_close', 'pct_diff_open',
            'pct_diff_high', 'pct_diff_low', 'pct_diff_close',
            'k', 'd', 'rsi'])

        # Add current value columns
        result['open'] = self.values()['open']
        result['high'] = self.values()['high']
        result['low'] = self.values()['low']
        result['close'] = self.values()['close']

        # Add columns of differences
        result['num_diff_open'] = num_difference['open']
        result['num_diff_high'] = num_difference['high']
        result['num_diff_low'] = num_difference['low']
        result['num_diff_close'] = num_difference['close']
        result['pct_diff_open'] = pct_difference['open']
        result['pct_diff_high'] = pct_difference['high']
        result['pct_diff_low'] = pct_difference['low']
        result['pct_diff_close'] = pct_difference['close']
        result['pct_diff_volume'] = pct_difference['volume']

        # Add date related columns
        result['day'] = self.dates().day
        result['weekday'] = self.dates().weekday
        result['week'] = self.dates().week
        result['month'] = self.dates().month
        result['quarter'] = self.dates().quarter
        result['dayofyear'] = self.dates().dayofyear

        # Calculate the Stochastic values
        stochastic = math.Stochastic(self.values(), window=self._kwindow)
        result['k'] = stochastic.k()
        result['d'] = stochastic.d(window=self._dwindow)

        # Calculate the Miscellaneous values
        miscellaneous = math.Misc(self.values())
        result['rsi'] = miscellaneous.rsi(window=self._rsiwindow)

        # Delete the first row of the dataframe as it has NaN values from the
        # .diff() and .pct_change() operations
        result = result.iloc[self._ignore_row_count:]

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
        _result = pd.to_datetime(pd.DataFrame({
            'year': self.dates().year.values.tolist(),
            'month': self.dates().month.values.tolist(),
            'day': self.dates().day.values.tolist()})).values
        result = _result[self._ignore_row_count:]

        # Return
        return result


class DataFile(DataSource):
    """Class ingests file data."""

    def __init__(self, filename):
        """Intialize the class.

        Args:
            filename: Name of file
            symbol: Symbol to update

        Returns:
            None

        """
        # Setup inheritance
        DataSource.__init__(self)

        # Initialize key variables
        self._filename = filename

        # Get data from file
        (self._values, self._dates) = self._data()

    def _data(self):
        """Process file data.

        Args:
            None

        Returns:
            result: (_values, _data) Pandas DataFrame tuple of values and dates

        """
        # Read data
        headings = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        data = pd.read_csv(self._filename, names=headings)
        data = data.drop(['time'], axis=1)

        # Drop date column from data
        _values = data.drop(['date'], axis=1)

        # Get date values from data
        _dates = general.Dates(data['date'], '%Y.%m.%d')

        # Return
        return (_values, _dates)

    def values(self):
        """Process file data.

        Args:
            None

        Returns:
            result: DataFrame of values

        """
        # Return
        result = self._values
        return result

    def dates(self):
        """Process file data.

        Args:
            None

        Returns:
            result: DataFrame of dates

        """
        # Return
        result = self._dates
        return result


class DataGRU(DataFile):
    """Super class for file data ingestion."""

    def __init__(self, filename, shift_steps):
        """Intialize the class.

        Args:
            filename: Name of file

        Returns:
            None

        """
        # Setup inheritance
        DataFile.__init__(self, filename)

        # Initialize key variables
        self._shift_steps = shift_steps
        self._label2predict = 'close'

        # Process data
        (self._vectors, self._classes) = self._vector_targets()

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

    def vectors(self):
        """Create a numpy array of vectors.

        Args:
            None

        Returns:
            result: Tuple of numpy array of vectors. (train, test)

        """
        # Return
        result = (self._vectors['train'], self._vectors['all'])
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
        label2predict = 'close'
        x_data = {'train': None, 'all': None}
        y_data = {'train': None, 'all': None}
        desired_columns = [
            'open', 'high', 'low', 'close',
            'weekday', 'day', 'dayofyear', 'quarter', 'month', 'num_diff_open',
            'num_diff_high', 'num_diff_low', 'num_diff_close', 'pct_diff_open',
            'pct_diff_high', 'pct_diff_low', 'pct_diff_close',
            'k', 'd', 'rsi']

        # Remove all undesirable columns from the dataframe
        pandas_df = self.dataframe()
        imported_columns = list(pandas_df)
        for column in imported_columns:
            if column not in desired_columns:
                pandas_df.drop(column, axis=1)

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
