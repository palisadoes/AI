"""Library to process the ingest of data files."""

# Standard imports
from __future__ import print_function
from copy import deepcopy
import sys
import os
import pickle
import time
import multiprocessing

# PIP imports
import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume
from statsmodels.graphics.tsaplots import plot_acf

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

import lightgbm as lgb

# Append custom application libraries
from merlin import general
from merlin import math


class DataIngest(object):
    """Parent class for ingesting data."""

    def __init__(self):
        """Intialize the class.

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        pass


class DataFile(DataIngest):
    """Class ingests data from files."""

    def __init__(self, filename):
        """Intialize the class.

        Args:
            filename: Name of file
            symbol: Symbol to update

        Returns:
            None

        """
        # Refer to parent class
        DataIngest.__init__(self)

        # Initialize key variables
        self._filename = filename

        # Get data from file
        (self._df_values, self._df_dates) = self._df_data()

    def _df_data(self):
        """Ingest data from file.

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
        """Create dataframe with open, high, low, close, and volume columns.

        Args:
            None

        Returns:
            result: DataFrame of values

        """
        # Return
        result = self._df_values
        return result

    def dates(self):
        """Create dataframe of dates from file.

        Args:
            None

        Returns:
            result: DataFrame of dates

        """
        # Return
        result = self._df_dates
        return result


class Data(object):
    """Class for processing data retrieval.

    The primary aim of this class is to create training, validation and test
    data for use by other machine learning classes.

    """

    def __init__(
            self, dataobject, shift_steps, test_size=0.1, binary=False):
        """Intialize the class.

        Args:
            dataobject: DataIngest object
            shift_steps: List of time-steps to shift the target-data to create
                future expected series values for prediction.
            test_size: Fraction of dataset to be used for test data
            binary: Process data for predicting boolean up / down movement vs
                actual values if True

        Returns:
            None

        """
        # Initialize key variables
        self._binary = bool(binary)
        self._shift_steps = shift_steps
        self._test_size = test_size

        if bool(self._binary) is False:
            self._label2predict = 'close'
        else:
            self._label2predict = 'increasing'

        # Get data from the DataIngest
        self._ohlcv = dataobject.ohlcv()
        self._dates = dataobject.dates()

        # Setup classwide variables to be used by indicators
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

        '''
        Define the number of rows at the beginning of the master dataframe to
        be ignored. Many of the indicator functions create NaN values at the
        beginning of the pd.Series they return. These NaNs in the data can
        cause errors and therefore must be removed.
        '''
        self._ignore_row_count = max(
            1,
            self._globals['kwindow'] + self._globals['dwindow'],
            max(self._globals.values()))

        # Sentiment values used by some indicator functions
        self._buy = 1
        self._sell = -1
        self._hold = 0
        self._strong = 1
        self._weak = 0

        # Create the master dataframes to be used by all other methods
        (self._dataframe, self._dataclasses) = self.__dataframe()

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
        """Plot autocorrelation of data.

        Source -
            https://machinelearningmastery.com/feature-selection-time-series-forecasting-python/

        Args:
            None

        Returns:
            None: Series for learning

        """
        # Get training vectors and classes
        (_, classes) = self._training_vectors_classes()

        # Convert the zeroth column of classes to a 1d np.array
        classes_1d = classes.values[:, 0]

        # Do the plotting
        plot_acf(classes_1d)
        plt.show()

    def feature_importance(self):
        """Plot feature importance of data.

        Source -
            https://machinelearningmastery.com/feature-selection-time-series-forecasting-python/

        Args:
            None

        Returns:
            None: Series for learning

        """
        # Get training vectors and classes
        (vectors, classes) = self._training_vectors_classes()

        # Convert the zeroth column of classes to a 1d np.array
        classes_1d = classes.values[:, 0]

        # Fit random forest model
        estimator = self._estimator()
        estimator.fit(vectors.values, classes_1d)

        # Show importance scores
        print('> Feature Importances:\n')
        print(estimator.feature_importances_)

        # Plot importance scores
        names = vectors.columns.values[:]
        ticks = [i for i in range(len(names))]
        plt.title('Feature Importance')
        plt.bar(ticks, estimator.feature_importances_)
        plt.xticks(ticks, names, rotation=-90)
        plt.show()

    def suggested_features(self, count=4, display=False):
        """Get a suggested list of features to use.

        Source -
            https://machinelearningmastery.com/feature-selection-time-series-forecasting-python/

        Args:
            None

        Returns:
            features: list of features to use.

        """
        # Initialize key variables
        features = []
        ts_start = time.time()
        filename = os.path.expanduser('/tmp/selection.pickle')
        ranking_tuple_list = []

        # Get training vectors and classes
        (vectors, classes) = self._training_vectors_classes()

        # Drop highly correlated columns
        columns = general.correlated_columns(vectors, threshold=0.95)
        vectors = vectors.drop(columns, axis=1)
        print('> Correlated Columns to Drop: {}'.format(columns))

        # Convert the zeroth column of classes to a 1d np.array
        classes_1d = classes.values[:, 0]

        # Print status
        print('> Calculating Suggested Features')

        # Perform feature selection
        if os.path.exists(filename) is True:
            fit = pickle.load(open(filename, 'rb'))
        else:
            estimator = self._estimator()
            selector = RFE(estimator, count)

            # Fit the data to the model
            fit = selector.fit(vectors.values, classes_1d)

            # Save to file
            pickle.dump(fit, open(filename, 'wb'))

        # More status
        print(
            '> Suggested Features Calculation duration: {0:.2f}s'
            ''.format(time.time() - ts_start))

        # Report selected features
        dataframe_header = vectors.columns.values[:]
        for i in range(len(fit.support_)):
            if fit.support_[i]:
                feature = dataframe_header[i]
                features.append(feature)

        # Print results
        for index in range(len(dataframe_header)):
            ranking_tuple_list.append(
                (dataframe_header[index], fit.ranking_[index])
            )
        ranking_tuple_list = sorted(ranking_tuple_list, key=lambda x: x[1])
        print('> Selected Features (Feature, Ranking Value):')
        for item in ranking_tuple_list:
            print('\t{:30} {:.3f}'.format(item[0], item[1]))

        # Plot feature rank
        if bool(display) is True:
            ticks = [i for i in range(len(dataframe_header))]
            plt.title('Suggested Features (Lower Values Better)')
            plt.bar(ticks, fit.ranking_)
            plt.xticks(ticks, dataframe_header, rotation=-90)
            plt.show()
            plt.close()

        # Plot feature importances
        if bool(display) is True:
            ticks = [i for i in range(len(ranking_tuple_list))]
            sorted_headings = [i[0] for i in ranking_tuple_list]
            sorted_features = [i[1] for i in ranking_tuple_list]
            plt.title('Suggested Features - Sorted (Lower Values Better)')
            plt.bar(ticks, sorted_features)
            plt.xticks(ticks, sorted_headings, rotation=-90)
            plt.show()
            plt.close()

        # Returns
        return features

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

    def _estimator(self):
        """Construct a gradient boosting estimator model.

        Source -
            https://machinelearningmastery.com/feature-selection-time-series-forecasting-python/

        Args:
            None

        Returns:
            estimator: A supervised learning estimator with a fit method that
                provides information about feature importance either through a
                coef_ attribute or through a feature_importances_ attribute.

        """
        # Initialize key variables
        cpu_cores = multiprocessing.cpu_count()
        n_estimators = 500

        # Use LightGBM with GPU
        params = {'device': 'gpu'}

        # Generate model
        if self._binary is False:
            objective = 'regression'
        else:
            params['num_class'] = len(self._shift_steps)
            objective = 'binary'

        estimator = lgb.LGBMRegressor(
            objective=objective,
            n_estimators=n_estimators,
            n_jobs=cpu_cores - 2)
        estimator.set_params(**params)

        # Return
        return estimator

    def __dataframe(self):
        """Create an comprehensive list of  from data.

        Args:
            None

        Returns:
            result: dataframe for learning

        """
        # Calculate the percentage and real differences between columns
        difference = math.Difference(self._ohlcv)
        num_difference = difference.actual()
        pct_difference = difference.relative()

        # Create result to return.
        result = pd.DataFrame()

        # Add current value columns
        # NOTE Close must be first for correct correlation column dropping
        result['close'] = self._ohlcv['close']
        result['open'] = self._ohlcv['open']
        result['high'] = self._ohlcv['high']
        result['low'] = self._ohlcv['low']
        result['volume'] = self._ohlcv['volume']

        # Add columns of differences
        # NOTE Close must be first for correct correlation column dropping
        result['num_diff_close'] = num_difference['close']
        result['pct_diff_close'] = pct_difference['close']

        result['num_diff_open'] = num_difference['open']
        result['pct_diff_open'] = pct_difference['open']

        result['num_diff_high'] = num_difference['high']
        result['pct_diff_high'] = pct_difference['high']

        result['num_diff_low'] = num_difference['low']
        result['pct_diff_low'] = pct_difference['low']
        result['pct_diff_volume'] = pct_difference['volume']

        # Add date related columns
        # result['day'] = self._dates.day
        result['weekday'] = self._dates.weekday
        # result['week'] = self._dates.week
        result['month'] = self._dates.month
        result['quarter'] = self._dates.quarter
        # result['dayofyear'] = self._dates.dayofyear

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
        result['vol_amplitude'] = abs(_min - _max)

        _min = result['volume'].rolling(
            2 * self._globals['week']).min()
        _max = result['volume'].rolling(
            2 * self._globals['week']).max()
        result['vol_amplitude_long'] = abs(_min - _max)

        # Volume metrics
        result['force_index'] = volume.force_index(
            result['close'], result['volume'])
        result['negative_volume_index'] = volume.negative_volume_index(
            result['close'], result['volume'])
        result['ease_of_movement'] = volume.ease_of_movement(
            result['high'], result['low'], result['close'], result['volume'])
        result['acc_dist_index'] = volume.acc_dist_index(
            result['high'], result['low'], result['close'], result['volume'])
        result['on_balance_volume'] = volume.on_balance_volume(
            result['close'], result['volume'])
        result['on_balance_volume_mean'] = volume.on_balance_volume(
            result['close'], result['volume'])
        result['volume_price_trend'] = volume.volume_price_trend(
            result['close'], result['volume'])

        # Calculate the Stochastic values
        result['k'] = momentum.stoch(
            result['high'],
            result['low'],
            result['close'],
            n=self._globals['kwindow'])

        result['d'] = momentum.stoch_signal(
            result['high'],
            result['low'],
            result['close'],
            n=self._globals['kwindow'],
            d_n=self._globals['dwindow'])

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
        _increasing = (_result >= 0).astype(int) * self._buy
        _decreasing = (_result < 0).astype(int) * self._sell
        result['increasing'] = _increasing + _decreasing

        # Stochastic subtraciton
        result['k_d'] = pd.Series(result['k'].values - result['d'].values)

        # Other indicators
        result['k_i'] = self._stochastic_indicator(
            result['k'], result['high'], result['low'], result['ma_close'])
        result['d_i'] = self._stochastic_indicator(
            result['d'], result['high'], result['low'], result['ma_close'])
        result['stoch_i'] = self._stochastic_indicator_2(
            result['k'], result['d'],
            result['high'], result['low'], result['ma_close'])
        result['rsi_i'] = self._rsi_indicator(
            result['rsi'], result['high'], result['low'], result['ma_close'])
        result['adx_i'] = self._adx_indicator(result['adx'])
        result['macd_diff_i'] = self._macd_diff_indicator(result['macd_diff'])
        result['volume_i'] = self._volume_indicator(
            result['ma_volume'], result['ma_volume_long'])

        # Create time shifted columns
        for step in range(1, self._ignore_row_count + 1):
            result['t-{}'.format(step)] = result['close'].shift(step)
            result['tpd-{}'.format(step)] = result[
                'close'].pct_change(periods=step)
            result['tad-{}'.format(step)] = result[
                'close'].diff(periods=step)

        # Mask increasing with
        result['increasing_masked'] = _mask(
            result['increasing'].to_frame(),
            result['stoch_i'],
            as_integer=True).values

        # Get class values for each vector
        classes = pd.DataFrame(columns=self._shift_steps)
        for step in self._shift_steps:
            # Shift each column by the value of its label
            if self._binary is True:
                # Classes need to be 0 or 1 (One hot encoding)
                classes[step] = (
                    result[self._label2predict].shift(-step) > 0).astype(int)
            else:
                classes[step] = result[self._label2predict].shift(-step)
            # classes[step] = result[self._label2predict].shift(-step)

        # Delete the firsts row of the dataframe as it has NaN values from the
        # .diff() and .pct_change() operations
        ignore = max(max(self._shift_steps), self._ignore_row_count)
        result = result.iloc[ignore:]
        classes = classes.iloc[ignore:]

        # Convert result to float32 to conserve memory
        result = result.astype(np.float32)

        # Return
        return result, classes

    def _training_vectors_classes(self):
        """Create vectors and classes used for training feature selection.

        Args:
            None

        Returns:
            result: Tuple of training (vectors, classes)

        """
        # Split into input and output
        _vectors = self._dataframe
        _classes = self._dataclasses

        # Get rid of the NaNs in the vectors and classes.
        (nanless_vectors, nanless_classes) = _no_nans(
            _vectors, _classes, self._shift_steps)

        # Work only with training data
        (vectors, _, __,
         classes, ___, ____) = general.train_validation_test_split(
             nanless_vectors, nanless_classes, self._test_size)

        # Return
        result = (vectors, classes)
        return result

    def _stochastic_indicator(self, s_value, high, low, _ma_close):
        """Create stochastic indicator.

        Return a pd.Series where the values are:
            -1 : Close is higher than the moving average AND
                the s_value > upper_limit
            1  : Close is lower than the moving average AND
                the s_value < lower_limit
            0  : All other cases

        Args:
            s_value: Either a stochastic K or D value pd.Series
            high: High value pd.Series
            low: Low value pd.Series
            _ma_close: pd.Series moving average of the close

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        upper_limit = 90
        lower_limit = 10
        ma_close = _ma_close.fillna(0).values

        # Sell criteria
        high_gt_ma_close = (high.values > ma_close).astype(int)
        sell_indicator = (s_value > upper_limit).astype(int) * self._sell
        sell = sell_indicator * high_gt_ma_close

        # Buy criteria
        low_lt_ma_close = (low.values < ma_close).astype(int)
        buy_indicator = (s_value < lower_limit).astype(int) * self._buy
        buy = buy_indicator * low_lt_ma_close

        # Return
        result = (buy + sell).astype(int)
        return result

    def _stochastic_indicator_2(
            self, _k_series, _d_series, high, low, _ma_close):
        """Create stochastic indicator.

        Return a pd.Series where the values are:
            -1 : Close is higher than the moving average AND
                    the stochastic D (moving average) > stochastic K AND
                    the stochastic D (moving average) > upper_limit
            1 : Close is lower than the moving average AND
                    the stochastic D (moving average) < stochastic K AND
                    the stochastic D (moving average) < lower_limit
            0  : All other cases

        Args:
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

        # Convert NaNs to zeros to prevent runtime errors
        k_series = _k_series.fillna(0)
        d_series = _d_series.fillna(0)
        ma_close = _ma_close.fillna(0).values

        # Sell criteria
        high_gt_ma_close = (high.values > ma_close).astype(int)
        d_gt_k = (d_series.values > k_series.values).astype(int)
        d_gt_upper = (d_series.values > upper_limit).astype(int)
        d_in_sell_zone = d_gt_upper * d_gt_k
        sell = (d_in_sell_zone * high_gt_ma_close) * self._sell

        # Buy criteria
        low_lt_ma_close = (low.values < ma_close).astype(int)
        k_gt_d = (d_series.values < k_series.values).astype(int)
        d_lt_lower = (d_series.values < lower_limit).astype(int)
        d_in_buy_zone = d_lt_lower * k_gt_d
        buy = (d_in_buy_zone * low_lt_ma_close) * self._buy

        # Return
        result = (buy + sell).astype(int)
        return result

    def _rsi_indicator(self, _rsi_series, high, low, _ma_close):
        """Create rsi indicator.

        Return a pd.Series where the values are:
            -1 : Close is higher than the moving average AND
                the s_value > upper_limit
            1  : Close is lower than the moving average AND
                the s_value < lower_limit
            0  : All other cases

        Args:
            _rsi_series: pd.Series of RSI values
            high: High value pd.Series
            low: Low value pd.Series
            ma_close: pd.Series moving average of the close

        Returns:
            result: Numpy array for learning

        """
        # Initialize key variables
        upper_limit = 70
        lower_limit = 30
        ma_close = _ma_close.fillna(0).values

        # Convert NaNs to zeros to prevent runtime errors
        rsi_series = _rsi_series.fillna(0)

        # Sell criteria
        high_gt_ma_close = (high.values > ma_close).astype(int)
        sell_indicator = (
            rsi_series.values > upper_limit).astype(int) * self._sell
        sell = sell_indicator * high_gt_ma_close

        # Buy criteria
        low_lt_ma_close = (low.values < ma_close).astype(int)
        buy_indicator = (
            rsi_series.values < lower_limit).astype(int) * self._buy
        buy = buy_indicator * low_lt_ma_close

        # Return
        result = (buy + sell).astype(int)
        return result

    def _volume_indicator(self, _short, _long):
        """Give indication of sell or buy action.

        Based on whether the long and short volume moving averages cross.

        Return a pd.Series where the values are:

        -1 : Short > Long = Sell
        1  : Long > Short = Buy

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
        result = (buy + sell).astype(int)

        # Return
        return result

    def _adx_indicator(self, _adx):
        """Create ADX indicator.

        If the ADX value is > 25, then we have strong activity.

        Return a pd.Series where the values are:

        1 : When _adx > 25
        0 : When _adx <= 25

        Args:
            _adx: Current ADX pd.Series

        Returns:
            result: Numpy array for learning

        """
        # Evaluate decisions
        result = ((_adx > 25).astype(int) * self._strong).astype(int)

        # Return
        return result

    def _macd_diff_indicator(self, macd):
        """Create MACD difference variable as an indicator.

        Return a pd.Series where the values are:

        1 : When macd > 0
        0 : When macd <= 0

        Args:
            macd: Current MACD pd.Series

        Returns:
            result: Numpy array for learning

        """
        # Evaluate decisions
        buy = (macd > 0).astype(int) * self._buy
        sell = (macd <= 0).astype(int) * self._sell
        result = (buy + sell).astype(int)

        # Return
        return result


class DataGRU(Data):
    """Prepare data for use by GRU models."""

    def __init__(self, dataobject, shift_steps, test_size=0.1, binary=False):
        """Intialize the class.

        Args:
            dataobject: DataIngest object
            shift_steps: List of time-steps to shift the target-data to create
                future expected series values for prediction.
            test_size: Fraction of dataset to be used for test data
            binary: Process data for predicting boolean up / down movement vs
                actual values if True

        Returns:
            None

        """
        # Setup inheritance
        Data.__init__(
            self, dataobject, shift_steps, test_size=test_size, binary=binary)

        # Process data
        (self._vectors,
         self._classes,
         self._filtered_columns) = self._create_vector_classes()

    def vectors(self):
        """Get all vectors for testing.

        Args:
            None

        Returns:
            result: pd.DataFrame of all feature vectors

        """
        # Return
        return self._vectors['NoNaNs']

    def classes(self):
        """Get all classes for testing.

        Args:
            None

        Returns:
            result: pd.DataFrame of all classes

        """
        # Return
        return self._classes['NoNaNs']

    def vectors_test_all(self):
        """Get vectors for testing.

        Also include any NaN values there may have been due to shifting

        Args:
            None

        Returns:
            result: pd.DataFrame of all feature vectors

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

    def stochastic_vectors_classes(self):
        """Create test vectors and classes filtered by ['stoch_i'] values.

        Args:
            None

        Returns:
            result: Tuple (filtered vectors, corresponding filtered classes)

        """
        # Split into input and output
        _vectors = self._dataframe
        _classes = self._dataclasses

        # Get rid of the NaNs in the vectors and classes
        (vectors, classes) = _no_nans(_vectors, _classes, self._shift_steps)

        # Get test vectors
        (test_vectors, test_classes) = general.test_vectors_classes(
            vectors, classes, self._test_size)

        # Filter by stoch_i
        filtered_indices = test_vectors[
            np.round(test_vectors.stoch_i) != 0].index.values
        filtered_vectors = test_vectors.loc[filtered_indices]
        filtered_classes = test_classes.loc[filtered_indices]

        # Remove all undesirable columns from the dataframe
        dataframe = _filtered_dataframe(
            filtered_vectors, self._filtered_columns)

        # Return
        result = (dataframe.values, filtered_classes.values)
        return result

    def _create_vector_classes(self):
        """Create vectors and targets from data.

        Args:
            None

        Returns:
            result: dataframe for learning

        """
        # Initialize key variables
        classes = deepcopy(self._dataclasses)
        x_data = {'NoNaNs': None, 'all': None}
        y_data = {'NoNaNs': None, 'all': None}

        # Try automated feature selection
        filtered_columns = self.suggested_features(count=20)

        # Remove all undesirable columns from the dataframe
        dataframe = _filtered_dataframe(self._dataframe, filtered_columns)

        # Create class and vector dataframes with only non NaN values
        # (val_loss won't improve otherwise)
        (x_data['NoNaNs'], y_data['NoNaNs']) = _no_nans(
            dataframe, classes, self._shift_steps, to_series=True)
        y_data['all'] = classes.values[:]
        x_data['all'] = dataframe.values[:]

        # Return
        return (x_data, y_data, filtered_columns)

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


def _mask(dataframe, mask_series, as_integer=False):
    """Apply a mask to the dataframe.

    Args:
        dataframe: Dataframe
        mask_series: pd.Series to use for masking. Where the pd.Series value
            is non-zero, the mask to be applied is 1 (one), else it is
            0 (zero). Zero values are used to eliminate unwanted values via
            multiplication.

    Returns:
        result: numpy array of timestamps

    """
    # Initialize key variables
    headings = list(dataframe)
    result = deepcopy(dataframe)

    # Create mask
    mask = (np.array(mask_series) != 0).astype(int)

    # Apply mask
    for heading in headings:
        if as_integer is False:
            result[heading] = result[heading] * mask
        else:
            result[heading] = np.array(result[heading] * mask).astype(int)

    # Return
    return result


def _create_classes(series, shift_steps):
    """Create classes from pd.Series.

    Args:
        series: pd.Series of values to predict
        shift_steps: List of time-steps to shift the target-data to create
            future expected series values for prediction.

    Returns:
        result: numpy array of timestamps

    """
    # Get class values for each vector
    classes = pd.DataFrame(columns=shift_steps)
    for step in shift_steps:
        # Shift each column by the value of its label
        classes[step] = series.shift(-step)

    # Return
    return classes


def _no_nans(
        _vectors, _classes, shift_steps, to_series=False):
    """Trim classes and vector dataframes of NaN values due to future steps.

    Args:
        _vectors: pd.DataFrame of vectors
        _classes: pd.DataFrame of classes
        shift_steps: List of time-steps to shift the target-data to create
            future expected series values for prediction.

    Returns:
        result: numpy array of timestamps

    """
    # Initialize key variables
    crop_by = max(shift_steps)

    # Crop trailing rows
    classes = _classes[:-crop_by]
    vectors = _vectors[:-crop_by]

    # Create class and vector dataframes with only non NaN values
    # (val_loss won't improve otherwise)
    if bool(to_series) is True:
        classes = classes.values
        vectors = vectors.values

    # Return
    result = (vectors, classes)
    return result


def _filtered_dataframe(dataframe, filtered_columns):
    """Remove undesirable columns from a dataframe.

    Args:
        _dataframe: DataFrame to filter
        filtered_columns: Column names to use for filtering

    Returns:
        result: DataFrame for learning

    """
    # Initialize key variables
    result = deepcopy(dataframe)

    # Remove all undesirable columns from the dataframe
    imported_columns = list(result)
    for column in imported_columns:
        if column not in filtered_columns:
            result = result.drop(column, axis=1)

    # Return
    return result
