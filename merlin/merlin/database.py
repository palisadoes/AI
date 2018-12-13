"""Library to process the ingest of data files."""

# Standard imports
from __future__ import print_function
from copy import deepcopy
import sys
import multiprocessing

# PIP imports
import pandas as pd
import numpy as np
from ta import trend, momentum, volatility
from statsmodels.graphics.tsaplots import plot_acf

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

# Append custom application libraries
from merlin import general
from merlin import math


class DataFile(object):
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
        (self._df_values, self._df_dates) = self._df_data()

    def _df_data(self):
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

    def ohlcv(self):
        """Process file data.

        Args:
            None

        Returns:
            result: DataFrame of values

        """
        # Return
        result = self._df_values
        return result

    def dates(self):
        """Process file data.

        Args:
            None

        Returns:
            result: DataFrame of dates

        """
        # Return
        result = self._df_dates
        return result


class Data(object):
    """Super class handling data retrieval."""

    def __init__(self, dataobject, binary=False):
        """Intialize the class.

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        if bool(binary) is False:
            self._label2predict = 'close'
        else:
            self._label2predict = 'increasing'

        # Get data
        self._ohlcv = dataobject.ohlcv()
        self._dates = dataobject.dates()

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
            'week': 5,
            'proc_window': 5
        }
        self._ignore_row_count = max(
            1,
            self._globals['kwindow'] + self._globals['dwindow'],
            max(self._globals.values()))

        # Sentiment values
        self._buy = 1
        self._sell = -1
        self._hold = 0
        self._strong = 1
        self._weak = 0

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
        result = self._ohlcv['open'][
            self._ignore_row_count:-self._globals['proc_window']]
        return result

    def high(self):
        """Get high values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self._ohlcv['high'][
            self._ignore_row_count:-self._globals['proc_window']]
        return result

    def low(self):
        """Get low values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self._ohlcv['low'][
            self._ignore_row_count:-self._globals['proc_window']]
        return result

    def close(self):
        """Get close values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self._ohlcv['close'][
            self._ignore_row_count:-self._globals['proc_window']]
        return result

    def volume(self):
        """Get volume values.

        Args:
            None

        Returns:
            result: Series for learning

        """
        # Return
        result = self._ohlcv['volume'][
            self._ignore_row_count:-self._globals['proc_window']]
        return result

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

    def autocorrelation(self):
        """Plot autocorrelation.

        Source -
            https://machinelearningmastery.com/feature-selection-time-series-forecasting-python/

        Args:
            None

        Returns:
            None: Series for learning

        """
        # Do the plotting
        plot_acf(self.values())
        plt.show()

    def feature_importance(self):
        """Plot feature importance.

        Source -
            https://machinelearningmastery.com/feature-selection-time-series-forecasting-python/

        Args:
            None

        Returns:
            None: Series for learning

        """
        # Split into input and output
        data = self._dataframe
        vectors = data.values
        classes = self.values()

        # Fit random forest model
        model = RandomForestRegressor(n_estimators=500, random_state=1)
        model.fit(vectors, classes)

        # Show importance scores
        print('> Feature Importances:\n')
        print(model.feature_importances_)

        # Plot importance scores
        names = data.columns.values[:]
        ticks = [i for i in range(len(names))]
        plt.bar(ticks, model.feature_importances_)
        plt.xticks(ticks, names, rotation=-90)
        plt.show()

    def suggested_features(self, count=4, display=False):
        """Plot feature selection.

        Source -
            https://machinelearningmastery.com/feature-selection-time-series-forecasting-python/

        Args:
            None

        Returns:
            None: Series for learning

        """
        # initialize key variables
        features = []
        cpu_cores = multiprocessing.cpu_count()

        # Split into input and output
        data = self._dataframe
        vectors = data.values
        classes = self.values()

        # Perform feature selection
        rfe = RFE(RandomForestRegressor(
            n_estimators=500, random_state=1, n_jobs=cpu_cores - 2), count)
        fit = rfe.fit(vectors, classes)

        # Report selected features
        print('> Selected Features:')
        dataframe_header = data.columns.values[:]
        for i in range(len(fit.support_)):
            if fit.support_[i]:
                feature = dataframe_header[i]
                features.append(feature)
                print('\t', feature)

        # Plot feature rank
        if bool(display) is True:
            ticks = [i for i in range(len(dataframe_header))]
            plt.bar(ticks, fit.ranking_)
            plt.xticks(ticks, dataframe_header, rotation=-90)
            plt.show()

        # Returns
        return features

    def __dataframe(self):
        """Create vectors from data.

        Args:
            shift_steps: List of steps

        Returns:
            result: dataframe for learning

        """
        # Calculate the percentage and real differences between columns
        difference = math.Difference(self._ohlcv)
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
            'ma_open', 'ma_high', 'ma_low', 'ma_close', 'ma_std_close',
            'ma_volume', 'ma_volume_long',
            'amplitude', 'amplitude_medium', 'amplitude_long',
            'k_i', 'd_i', 'rsi_i', 'adx_i', 'macd_diff_i', 'volume_i',
            'std_pct_diff_close',
            'volume_amplitude', 'volume_amplitude_long',
            'bollinger_lband', 'bollinger_hband', 'bollinger_lband_indicator',
            'bollinger_hband_indicator',
            'ma_volume_delta']).astype('float16')

        # Add current value columns
        result['open'] = self._ohlcv['open']
        result['high'] = self._ohlcv['high']
        result['low'] = self._ohlcv['low']
        result['close'] = self._ohlcv['close']
        result['volume'] = self._ohlcv['volume']

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
        result['day'] = self._dates.day
        result['weekday'] = self._dates.weekday
        result['week'] = self._dates.week
        result['month'] = self._dates.month
        result['quarter'] = self._dates.quarter
        result['dayofyear'] = self._dates.dayofyear

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

        # Standard deviation related
        result['ma_std_close'] = result['close'].rolling(
            self._globals['ma_window']).std()
        result['std_pct_diff_close'] = result['pct_diff_close'].rolling(
            self._globals['ma_window']).std()
        result['bollinger_lband'] = volatility.bollinger_lband(result['close'])
        result['bollinger_hband'] = volatility.bollinger_lband(result['close'])
        result['bollinger_lband_indicator'] = volatility.bollinger_lband_indicator(result['close'])
        result['bollinger_hband_indicator'] = volatility.bollinger_hband_indicator(result['close'])

        # Rolling ranges
        result['amplitude'] = result['high'] - result['low']

        _min = result['low'].rolling(
            self._globals['week']).min()
        _max = result['high'].rolling(
            self._globals['week']).max()
        result['amplitude_medium'] = abs(_min - _max)

        _min = result['low'].rolling(
            2 * self._globals['week']).min()
        _max = result['high'].rolling(
            2 * self._globals['week']).max()
        result['amplitude_long'] = abs(_min - _max)

        _min = result['volume'].rolling(
            self._globals['week']).min()
        _max = result['volume'].rolling(
            self._globals['week']).max()
        result['volume_amplitude'] = abs(_min - _max)

        _min = result['volume'].rolling(
            2 * self._globals['week']).min()
        _max = result['volume'].rolling(
            2 * self._globals['week']).max()
        result['volume_amplitude_long'] = abs(_min - _max)

        # Calculate the Stochastic values
        stochastic = math.Stochastic(
            self._ohlcv, window=self._globals['kwindow'])
        result['k'] = stochastic.k()
        result['d'] = stochastic.d(window=self._globals['dwindow'])

        # Calculate the Miscellaneous values
        result['rsi'] = momentum.rsi(
            result['close'],
            n=self._globals['rsiwindow'],
            fillna=False)

        miscellaneous = math.Misc(self._ohlcv)
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

        # Other indicators
        result['k_i'] = self._stochastic_indicator(
            result['k'], result['high'], result['low'], result['ma_close'])
        result['d_i'] = self._stochastic_indicator(
            result['d'], result['high'], result['low'], result['ma_close'])
        result['rsi_i'] = self._rsi_indicator(
            result['rsi'], result['high'], result['low'], result['ma_close'])
        result['adx_i'] = self._adx_indicator(result['adx'])
        result['macd_diff_i'] = self._macd_diff_indicator(result['macd_diff'])
        result['volume_i'] = self._volume_indicator(
            result['ma_volume'], result['ma_volume_long'])

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
            'year': self._dates.year.values.tolist(),
            'month': self._dates.month.values.tolist(),
            'day': self._dates.day.values.tolist()})).values
        result = _result[self._ignore_row_count:]

        # Return
        return result

    def _stochastic_indicator(self, s_value, high, low, ma_close):
        """Create stochastic indicator.

        Args:
            selector:
                s_value: Either a stochastic K or D value pd.Series
                high: High value pd.Series
                low: Low value pd.Series
                ma_close: pd.Series moving average of the close

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        ma_close = ma_close.values

        # Sell criteria
        high_gt_ma_close = (
            high.values.astype(int) > ma_close.astype(int)).astype(int)
        sell_indicator = (s_value > 90).astype(int) * self._sell
        sell = sell_indicator * high_gt_ma_close

        # Buy criteria
        low_lt_ma_close = (
            low.values.astype(int) < ma_close.astype(int)).astype(int)
        buy_indicator = (s_value < 10).astype(int) * self._buy
        buy = buy_indicator * low_lt_ma_close

        # Return
        result = buy + sell
        return result

    def _stochastic_indicator_2(self, k_series, d_series, high, low, ma_close):
        """Create stochastic indicator.

        Args:
            selector:
                stochastic_value: Either a stochastic K or D value pd.Series
                high: High value pd.Series
                low: Low value pd.Series
                ma_close: pd.Series moving average of the close

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        upper_limit = 90
        lower_limit = 10
        ma_close = ma_close.values

        # Sell criteria
        high_gt_ma_close = (
            high.values.astype(int) > ma_close.astype(int)).astype(int)
        '''k_sell_indicator = (
            k_series.values > upper_limit).astype(int) * self._sell
        sell = k_sell_indicator * high_gt_ma_close'''

        d_gt_k = (d_series.values > k_series.values).astype(int)
        d_gt_upper = (d_series.values > upper_limit).astype(int)
        d_in_sell_zone = d_gt_upper * d_gt_k
        sell = (d_in_sell_zone * high_gt_ma_close) * self._sell

        # Buy criteria
        low_lt_ma_close = (
            low.values.astype(int) < ma_close.astype(int)).astype(int)
        '''k_buy_indicator = (
            k_series.values < lower_limit).astype(int) * self._buy
        buy = k_buy_indicator * low_lt_ma_close'''

        k_gt_d = (d_series.values < k_series.values).astype(int)
        d_lt_lower = (d_series.values < lower_limit).astype(int)
        d_in_buy_zone = d_lt_lower * k_gt_d
        buy = (d_in_buy_zone * low_lt_ma_close) * self._buy

        # Return
        result = buy + sell
        return result

    def _rsi_indicator(self, rsi_series, high, low, ma_close):
        """Create rsi indicator.

        Args:
            selector:
                rsi_series: Either a rsi K or D value pd.Series
                high: High value pd.Series
                low: Low value pd.Series
                ma_close: pd.Series moving average of the close

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        upper_limit = 70
        lower_limit = 30
        ma_close = ma_close.values

        # Sell criteria
        high_gt_ma_close = (
            high.values.astype(int) > ma_close.astype(int)).astype(int)
        sell_indicator = (
            rsi_series.values > upper_limit).astype(int) * self._sell
        sell = sell_indicator * high_gt_ma_close

        # Buy criteria
        low_lt_ma_close = (
            low.values.astype(int) < ma_close.astype(int)).astype(int)
        buy_indicator = (
            rsi_series.values < lower_limit).astype(int) * self._buy
        buy = buy_indicator * low_lt_ma_close

        # Return
        result = buy + sell
        return result

    def _volume_indicator(self, _short, _long):
        """Give indication of sell or buy action.

        Based on whether the long and short volume moving averages cross.

        Short > Long: Sell
        Long > Short: Buy

        Args:
            _short: Short term volume moving average pd.Series
            _long: Long term volume moving average pd.Series

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        _short = _short.values.astype(int)
        _long = _long.values.astype(int)

        # Create conditions for decisions
        sell = (_short > _long).astype(int) * self._sell
        buy = (_long >= _short).astype(int) * self._buy

        # Evaluate decisions
        result = buy + sell

        # Return
        return result

    def _adx_indicator(self, _adx):
        """Create ADX indicator.

        If the ADX value is > 25, then we have strong activity

        Args:
            _adx: Current ADX pd.Series

        Returns:
            result: Numpy array for learning

        """
        # Evaluate decisions
        result = (_adx > 25).astype(int) * self._strong

        # Return
        return result

    def _macd_diff_indicator(self, macd):
        """Create MACD difference variable as an indicator.

        Args:
            macd: Current MACD pd.Series

        Returns:
            result: Numpy array for learning

        """
        # Evaluate decisions
        buy = (macd > 0).astype(int) * self._buy
        sell = (macd < 0).astype(int) * self._sell
        result = buy + sell

        # Return
        return result


class DataGRU(Data):
    """Prepare data for use by GRU models."""

    def __init__(self, dataobject, shift_steps, test_size=0.1, binary=False):
        """Intialize the class.

        Args:
            filename: Name of file

        Returns:
            None

        """
        # Setup inheritance
        Data.__init__(self, dataobject, binary=binary)

        # Initialize key variables
        self._shift_steps = shift_steps
        self._test_size = test_size

        # Process data
        (self._vectors, self._classes) = self._create_vector_classes()

    def vectors(self):
        """Get all vectors for testing.

        Args:
            None

        Returns:
            result: Training or test vector numpy arrays

        """
        # Return
        return self._vectors['NoNaNs']

    def classes(self):
        """Get all vectors for testing.

        Args:
            None

        Returns:
            result: Training or test vector numpy arrays

        """
        # Return
        return self._classes['NoNaNs']

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
        pandas_df = deepcopy(self._dataframe)
        crop_by = max(self._shift_steps)
        label2predict = self._label2predict
        x_data = {'NoNaNs': None, 'all': None}
        y_data = {'NoNaNs': None, 'all': None}
        desired_columns = [
            'open', 'high', 'low', 'close',
            'weekday', 'day', 'dayofyear', 'quarter', 'month', 'num_diff_open',
            'num_diff_high', 'num_diff_low', 'num_diff_close', 'pct_diff_open',
            'pct_diff_high', 'pct_diff_low', 'pct_diff_close',
            'std_pct_diff_close', 'ma_std_close',
            'bollinger_lband', 'bollinger_hband', 'bollinger_lband_indicator',
            'bollinger_hband_indicator',
            'amplitude', 'amplitude_medium', 'amplitude_long',
            'k_i', 'd_i', 'rsi_i', 'adx_i', 'macd_diff_i', 'volume_i',
            'volume_amplitude', 'volume_amplitude_long',
            'k', 'd', 'rsi', 'adx', 'proc', 'macd_diff', 'ma_volume_delta']

        # Try automated feature selection
        desired_columns = self.suggested_features(count=10)

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
        y_data['NoNaNs'] = classes.values[:-crop_by]
        y_data['all'] = classes.values[:]
        x_data['NoNaNs'] = pandas_df.values[:-crop_by]
        x_data['all'] = pandas_df.values[:]

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


class DataDT(Data):
    """Prepare data for use by decision tree models."""

    def __init__(self, dataobject, actuals=False, steps=0):
        """Intialize the class.

        Args:
            filename: Name of file
            actuals: If True use actual values not 1, 0, -1 wherever possible

        Returns:
            None

        """
        # Setup inheritance
        Data.__init__(self, dataobject)

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
