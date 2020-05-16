"""Library to process the ingest of data files."""

# PIP imports
import pandas as pd
import numpy as np
from ta import trend, momentum

# Append custom application libraries
from merlin import general
from merlin import math


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

    def _file_values(self):
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


class DataSource(_DataFile):
    """Super class handling data retrieval."""

    def __init__(self, filename):
        """Intialize the class.

        Args:
            None

        Returns:
            None

        """
        # Setup inheritance
        _DataFile.__init__(self, filename)

        # Setup classwide variables
        self._globals = {
            'kwindow': 35,
            'dwindow': 5,
            'rsiwindow': 35,
            'ma_window': 13,
            'vma_window': 2,
            'vma_window_long': 14,
            'adx_window': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_sign': 9,
            'proc_window': 5
        }
        self._ignore_row_count = max(
            1,
            self._globals['kwindow'] + self._globals['dwindow'],
            max(self._globals.values()))

        # Create the dataframe to be used by all other methods
        self._dataframe = self.__dataframe()

    def open(self):
        """Get open values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self._file_values()['open'][self._ignore_row_count:]
        return result

    def high(self):
        """Get high values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self._file_values()['high'][self._ignore_row_count:]
        return result

    def low(self):
        """Get low values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self._file_values()['low'][self._ignore_row_count:]
        return result

    def close(self):
        """Get close values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self._file_values()['close'][self._ignore_row_count:]
        return result

    def volume(self):
        """Get volume values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self._file_values()['volume'][self._ignore_row_count:]
        return result

    def __dataframe(self):
        """Create vectors from data.

        Args:
            shift_steps: List of steps

        Returns:
            result: dataframe for learning

        """
        # Calculate the percentage and real differences between columns
        difference = math.Difference(self._file_values())
        num_difference = difference.actual()
        pct_difference = difference.relative()

        # Create result to return.
        # Make sure it is float16 for efficient computing
        result = pd.DataFrame(columns=[
            'open', 'high', 'low', 'close', 'volume', 'increasing',
            'weekday', 'day', 'dayofyear', 'quarter', 'month', 'num_diff_open',
            'num_diff_high', 'num_diff_low', 'num_diff_close', 'pct_diff_open',
            'pct_diff_high', 'pct_diff_low', 'pct_diff_close',
            'k', 'd', 'rsi', 'adx', 'macd_diff', 'proc',
            'ma_open', 'ma_high', 'ma_low', 'ma_close',
            'ma_volume', 'ma_volume_long',
            'ma_volume_delta']).astype('float16')

        # Add current value columns
        result['open'] = self._file_values()['open']
        result['high'] = self._file_values()['high']
        result['low'] = self._file_values()['low']
        result['close'] = self._file_values()['close']
        result['volume'] = self._file_values()['volume']

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

        # Moving averages
        result['ma_open'] = result['open'].rolling(
            self._globals['ma_window']).mean()
        result['ma_high'] = result['high'].rolling(
            self._globals['ma_window']).mean()
        result['ma_low'] = result['low'].rolling(
            self._globals['ma_window']).mean()
        result['ma_close'] = result['close'].rolling(
            self._globals['ma_window']).mean()
        result['ma_volume'] = result['volume'].rolling(
            self._globals['vma_window']).mean()
        result['ma_volume_long'] = result['volume'].rolling(
            self._globals['vma_window_long']).mean()
        result['ma_volume_delta'] = result[
            'ma_volume_long'] - result['ma_volume']

        # Calculate the Stochastic values
        stochastic = math.Stochastic(
            self._file_values(), window=self._globals['kwindow'])
        result['k'] = stochastic.k()
        result['d'] = stochastic.d(window=self._globals['dwindow'])

        # Calculate the Miscellaneous values
        result['rsi'] = momentum.rsi(
            result['close'],
            n=self._globals['rsiwindow'],
            fillna=False)

        miscellaneous = math.Misc(self._file_values())
        result['proc'] = miscellaneous.proc(self._globals['proc_window'])

        # Calculate ADX
        result['adx'] = trend.adx(
            result['high'],
            result['low'],
            result['close'],
            n=self._globals['adx_window'])

        # Calculate MACD difference
        result['macd_diff'] = trend.macd_diff(
            result['close'],
            n_fast=self._globals['macd_sign'],
            n_slow=self._globals['macd_slow'],
            n_sign=self._globals['macd_sign'])

        # Create series for increasing / decreasing closes (Convert NaNs to 0)
        _result = np.nan_to_num(result['pct_diff_close'].values)
        _increasing = (_result >= 0).astype(int) * 1
        _decreasing = (_result < 0).astype(int) * 0
        result['increasing'] = _increasing + _decreasing

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


class DataGRU(DataSource):
    """Prepare data for use by GRU models."""

    def __init__(self, filename, shift_steps, test_size=0.1, binary=False):
        """Intialize the class.

        Args:
            filename: Name of file

        Returns:
            None

        """
        # Setup inheritance
        DataSource.__init__(self, filename)

        # Initialize key variables
        self._shift_steps = shift_steps
        self._test_size = test_size
        if bool(binary) is False:
            self._label2predict = 'close'
        else:
            self._label2predict = 'increasing'

        # Process data
        (self._vectors, self._classes) = self._create_vector_classes()

    def values(self):
        """Get values that we are aiming to predict.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self._dataframe[self._label2predict]
        return result

    def vectors_test_all(self):
        """Get vectors for testing.

        Args:
            None

        Returns:
            result: Training or test vector numpy arrays

        """
        # Fraction of vectors to be used for training
        train_split = 1 - self._test_size

        # Total number of available vectors
        num_data = len(self._vectors['all'])

        # Fraction of vectors to be used for training and testing
        training_count = int(train_split * num_data)

        # Return
        result = self._vectors['all'][training_count:]
        return result

    def train_validation_test_split(self):
        """Create contiguous (not random) training and test data.

        train_test_split in sklearn.model_selection does this randomly and is
        not suited for time-series data. It also doesn't create a
        validation-set

        Args:
            None

        Returns:
            result: Training or test vector numpy arrays

        """
        # Return
        vectors = self._vectors['NoNaNs']
        classes = self._classes['NoNaNs']
        result = general.train_validation_test_split(
            vectors, classes, self._test_size)
        return result

    def _create_vector_classes(self):
        """Create vectors and targets from data.

        Args:
            shift_steps: List of steps

        Returns:
            result: dataframe for learning

        """
        # Initialize key variables
        pandas_df = self._dataframe
        crop_by = max(self._shift_steps)
        label2predict = self._label2predict
        x_data = {'NoNaNs': None, 'all': None}
        y_data = {'NoNaNs': None, 'all': None}
        desired_columns = [
            'open', 'high', 'low', 'close',
            'weekday', 'day', 'dayofyear', 'quarter', 'month', 'num_diff_open',
            'num_diff_high', 'num_diff_low', 'num_diff_close', 'pct_diff_open',
            'pct_diff_high', 'pct_diff_low', 'pct_diff_close',
            'k', 'd', 'rsi', 'adx', 'proc', 'macd_diff', 'ma_volume_delta']

        desired_columns = [
            'num_diff_open',
            'num_diff_high', 'num_diff_low', 'num_diff_close', 'pct_diff_open',
            'pct_diff_high', 'pct_diff_low', 'pct_diff_close']

        # Get class values for each vector
        classes = pd.DataFrame(columns=self._shift_steps)
        for step in self._shift_steps:
            # Shift each column by the value of its label
            classes[step] = pandas_df[label2predict].shift(-step)

        # Remove all undesirable columns from the dataframe
        imported_columns = list(pandas_df)
        for column in imported_columns:
            if column not in desired_columns:
                pandas_df = pandas_df.drop(column, axis=1)

        # Create class and vector dataframes with only non NaN values
        # (val_loss won't improve otherwise)
        y_data['NoNaNs'] = classes.values[:-crop_by].astype(np.float32)
        y_data['all'] = classes.values[:].astype(np.float32)
        x_data['NoNaNs'] = pandas_df.values[:-crop_by].astype(np.float32)
        x_data['all'] = pandas_df.values[:].astype(np.float32)

        # Return
        return(x_data, y_data)

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


class DataDT(DataSource):
    """Prepare data for use by decision tree models."""

    def __init__(self, filename, actuals=False, steps=0):
        """Intialize the class.

        Args:
            filename: Name of file
            actuals: If True use actual values not 1, 0, -1 wherever possible

        Returns:
            None

        """
        # Setup inheritance
        DataSource.__init__(self, filename)

        # Setup global variables
        self._steps = steps
        self._actuals = actuals
        self._buy = 1
        self._sell = -1
        self._hold = 0
        self._strong = 1
        self._weak = 0

    def classes(self):
        """Create volume variables.

        Args:
            None

        Returns:
            result: Numpy array of binary values for learning
                0 = sell
                1 = buy

        """
        # Initialize key variables
        close = self._dataframe['pct_diff_close']

        # Shift data and Trim last rows
        if bool(self._steps) is True:
            close = close.shift(-abs(self._steps))
            close = close.iloc[:-abs(self._steps)]

        # Convert DataFrame to Numpy array
        close = close.values
        buy = (close > 0).astype(int) * self._buy
        sell = (close < 0).astype(int) * self._sell
        _result = buy + sell

        # Reshape Numpy array
        rows = _result.shape[0]
        result = _result.reshape(rows, 1)

        # pandas_df[label2predict].shift(-step)

        # Return
        return result

    def vectors(self):
        """Create vector variables.

        Args:
            None

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        _result = pd.DataFrame(columns=[
            'k', 'd', 'rsi', 'volume', 'adx', 'proc', 'macd_diff'])
        _result['k'] = self._stochastic_k()
        _result['d'] = self._stochastic_d()
        _result['rsi'] = self._rsi()
        _result['volume'] = self._volume()
        _result['adx'] = self._adx()
        _result['macd_diff'] = self._macd_diff()
        _result['proc'] = self._proc()

        desired_columns = [
            'k', 'd', 'rsi', 'volume', 'adx', 'proc', 'macd_diff']

        # Remove all undesirable columns from the dataframe
        imported_columns = list(_result)
        for column in imported_columns:
            if column not in desired_columns:
                _result = _result.drop(column, axis=1)

        # Trim last row if necessary
        if bool(self._steps) is True:
            _result = _result.iloc[:-abs(self._steps)]

        # Return
        result = np.asarray(_result)
        return result

    def _stochastic_k(self):
        """Create stochastic K variables.

        Args:
            None

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        result = self._stochastic(True)

        # Return
        return result

    def _stochastic_d(self):
        """Create stochastic D variables.

        Args:
            None

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        result = self._stochastic(False)

        # Return
        return result

    def _stochastic(self, selector=None):
        """Create stochastic D variables.

        Args:
            selector:
                True = Calculate stochastic K values
                False = Calculate stochastic D values
                None = Calculate RSI values

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        if selector is True:
            s_value = self._dataframe['k'].values
        elif selector is False:
            s_value = self._dataframe['d'].values
        else:
            s_value = self._dataframe['rsi'].values

        if self._actuals is False:
            ma_close = self._dataframe['ma_close'].values
            high_gt_ma_close = self._dataframe['high'].values > ma_close
            low_lt_ma_close = self._dataframe['low'].values < ma_close

            # Create conditions for decisions
            sell = (s_value > 90).astype(int) * self._sell
            buy = (s_value < 10).astype(int) * self._buy

            # Condition when candle is straddling the moving average
            straddle = np.logical_and(
                high_gt_ma_close, low_lt_ma_close).astype(int)

            # Multiply the straddle
            result = straddle * (sell + buy)
        else:
            result = s_value

        # Return
        return result

    def _rsi(self):
        """Create RSI variables.

        Args:
            None

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        result = self._stochastic(None)

        # Return
        return result

    def _volume(self):
        """Create volume variables.

        Args:
            None

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        _short = self._dataframe['ma_volume'].values
        _long = self._dataframe['ma_volume_long'].values

        # Create conditions for decisions
        sell = (_short > _long).astype(int) * self._sell
        buy = (_long >= _short).astype(int) * self._buy

        # Evaluate decisions
        result = buy + sell

        # Return
        return result

    def _adx(self):
        """Create ADX variables.

        Args:
            None

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        _adx = self._dataframe['adx'].values

        # Evaluate decisions
        if self._actuals is False:
            result = (_adx > 25).astype(int) * self._strong
        else:
            result = _adx

        # Return
        return result

    def _macd_diff(self):
        """Create MACD difference variables.

        Args:
            None

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        _result = self._dataframe['macd_diff'].values

        # Evaluate decisions
        if self._actuals is False:
            buy = (_result > 0).astype(int) * self._buy
            sell = (_result < 0).astype(int) * self._sell
            result = buy + sell
        else:
            result = _result

        # Return
        return result

    def _proc(self):
        """Create PROC (Price Rate of Change) values for learning.

        Args:
            None

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        result = self._dataframe['proc'].values

        # Return
        return result
