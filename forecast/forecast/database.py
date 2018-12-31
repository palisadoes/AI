"""Library to process the ingest of data files."""

# Standard imports
import os
import pickle
import time
import csv
from copy import deepcopy
import multiprocessing
from datetime import datetime

# PIP imports
from sklearn.feature_selection import RFE
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import lightgbm as lgb

# Forecast imports
from forecast import general


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
    """Class ingests file data."""

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

    def data(self, ts_start=None, rrd_step=300):
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
            binary: Process data for predicting boolean up / down movement vs
                actual values if True

        Returns:
            None

        """
        # Initialize key variables
        self._shift_steps = shift_steps
        self._test_size = test_size
        self._binary = bool(binary)
        if self._binary is False:
            self._label2predict = 'value'
        else:
            self._label2predict = 'increasing'
        self._dataobject = dataobject

        # Create the master dataframes to be used by all other methods
        (self._dataframe, self._dataclasses) = self.__dataframe()

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

    def _training_vectors_classes(self):
        """Create vectors and classes used for training feature selection.

        Args:
            None

        Returns:
            result: Tuple of training (vectors, classes)

        """
        # Split into input and output
        _vectors = self._dataframe.drop(self._label2predict, axis=1)
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
        """Create vectors from data.

        Args:
            shift_steps: List of steps

        Returns:
            result: dataframe for learning

        """
        # Initialize key variables
        prior_period_count = 20
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
        result = pd.DataFrame()

        # Get list of values
        for epoch, value in sorted(self._dataobject.data().items()):
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

        # Create series for increasing / decreasing closes (Convert NaNs to 0)
        _result = np.nan_to_num(pd.Series(output['value']).pct_change().values)
        _increasing = (_result >= 0).astype(int) * 1
        _decreasing = (_result < 0).astype(int) * 0
        result['increasing'] = _increasing + _decreasing

        # Add current value columns
        result['value'] = pd.Series(output['value'])
        result['day'] = pd.Series(output['day'])
        result['weekday'] = pd.Series(output['weekday'])
        result['year'] = pd.Series(output['year'])
        result['hour'] = pd.Series(output['hour'])
        result['month'] = pd.Series(output['month'])
        result['minute'] = pd.Series(output['minute'])
        result['second'] = pd.Series(output['second'])
        # result['timestamp'] = pd.Series(output['timestamp'])

        # Create time shifted columns
        for step in range(1, prior_period_count + 1):
            result['t-{}'.format(step)] = result['value'].shift(step)
            result['tpd-{}'.format(step)] = result[
                'value'].pct_change(periods=step)
            result['tad-{}'.format(step)] = result[
                'value'].diff(periods=step)

        # Get class values for each vector
        classes = pd.DataFrame(columns=self._shift_steps)
        for step in self._shift_steps:
            # Shift each column by the value of its label
            classes[step] = result[self._label2predict].shift(-step)

        # Delete the first row of the dataframe as it has NaN values from the
        # .diff() and .pct_change() operations
        ignore = max(max(self._shift_steps), prior_period_count)
        result = result.iloc[ignore:]
        classes = classes.iloc[ignore:]

        # Convert result to float32 to conserve memory
        result = result.astype(np.float32)

        # Return
        return result, classes


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
        (self._vectors, self._classes) = self._create_vector_classes()

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
        classes = self._dataclasses
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
        return (x_data, y_data)


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
