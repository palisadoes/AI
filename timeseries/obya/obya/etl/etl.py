"""Module to process data files."""

from collections import namedtuple

# PIP3 package imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class Data():
    """Class for processing data."""

    def __init__(self, df_, shift=10):
        """Intialize the class.

        Args:
            df_: Pandas dataframe with 'timestamp' and 'value' columns
            shift: Number of steps to shift data to create vectors

        Returns:
            None

        """
        # Initialize key variables
        self._shift = shift
        self._df = df_

    def vectors(self):
        """Create vectors.

        Vectors are a time shift of the instantiated dataframe. Each column
        representing a shift in the past.

        Args:
            None

        Returns:
            dataframe: Vector dataframe, excluding NaNs created by shift

        """
        # Create vectors
        dataframe = pd.DataFrame()
        for index in range(self._shift, 0, -1):
            dataframe['t-{}'.format(index)] = self._df['value'].shift(index)
        dataframe['value'] = self._df.values
        return dataframe[self._shift:]

    def split(self, test_size=0.2):
        """Split vectors into training, test and validation sets.

        Args:
            test_size: If float, should be between 0.0 and 1.0 and represent
                the proportion of the dataset to include in the test split. If
                int, represents the absolute number of test samples.

        Returns:
            result: NamedTuple with the following attributes:
                'x_train, x_test, x_validate, y_train, y_test, y_validate'

        """
        # Initialize key variables
        Splits = namedtuple(
            'Splits',
            'x_train, x_test, x_validate, y_train, y_test, y_validate')
        xy_ = _xy(self.vectors())

        # X-Vectors are all columns except ['value']
        # ['value'] is always the right most column
        (x_train, x_later, y_train, y_later) = train_test_split(
            xy_.feature_vectors,
            xy_.value_vectors,
            test_size=test_size,
            shuffle=False)
        (x_test, x_validate, y_test, y_validate) = train_test_split(
            x_later,
            y_later,
            test_size=0.5,
            shuffle=False)

        result = Splits(
            x_train=x_train, x_test=x_test, x_validate=x_validate,
            y_train=y_train, y_test=y_test, y_validate=y_validate)
        return result

    def scaled_split(self):
        """Scale the training, test and validation sets.

        Args:
            None.

        Returns:
            result: NamedTuple with the following attributes:
                'x_train, x_test, x_validate, y_train, y_test, y_validate'

        """
        # Initialize key variables
        ScaledSplits = namedtuple(
            'ScaledSplits',
            'x_train, x_test, x_validate, y_train, y_test, y_validate')
        splits = self.split()
        xy_ = _xy(self.vectors())

        '''
        The neural network works best on values roughly between -1 and 1, so we
        need to scale the data before it is being input to the neural network.
        We can use scikit-learn for this.

        We first create a scaler-object for the input-signals.

        Then we detect the range of values from the training-data and scale
        the training-data.

        From StackOverflow:

        To center the data (make it have zero mean and unit standard error),
        you subtract the mean and then divide the result by the standard
        deviation.

            x'=x−μσ

        You do that on the training set of data. But then you have to apply the
        same transformation to your testing set (e.g. in cross-validation), or
        to newly obtained examples before forecast. But you have to use the
        same two parameters μ and σ (values) that you used for centering the
        training set.

        Hence, every sklearn's transform's fit() just calculates the parameters
        (e.g. μ and σ in case of StandardScaler) and saves them as an internal
        objects state. Afterwards, you can call its transform() method to apply
        the transformation to a particular set of examples.

        fit_transform() joins these two steps and is used for the initial
        fitting of parameters on the training set x, but it also returns a
        transformed x'. Internally, it just calls first fit() and then
        transform() on the same data.
        '''
        x_scaler = MinMaxScaler()
        _ = x_scaler.fit_transform(xy_.feature_vectors)

        '''
        The target-data comes from the same data-set as the input-signals,
        because it is the weather-data for one of the cities that is merely
        time-shifted. But the target-data could be from a different source with
        different value-ranges, so we create a separate scaler-object for the
        target-data.
        '''

        y_scaler = MinMaxScaler()
        _ = y_scaler.fit_transform(xy_.value_vectors)

        # Return
        result = ScaledSplits(
            x_train=x_scaler.transform(splits.x_train),
            x_test=x_scaler.transform(splits.x_test),
            x_validate=x_scaler.transform(splits.x_validate),
            y_train=y_scaler.transform(splits.y_train),
            y_test=y_scaler.transform(splits.y_test),
            y_validate=y_scaler.transform(splits.y_validate))
        return result


def _xy(vectors):
    """Create vectors and values.

    Args:
        None.

    Returns:
        result: NamedTuple with the following attribute DataFrames:
            'vectors, values'

    """
    # Initialize key variables
    VectorsValues = namedtuple(
        'VectorsValues', 'feature_vectors, value_vectors')

    # X-Vectors are all columns except ['value']
    # ['value'] is always the right most column
    result = VectorsValues(
        feature_vectors=vectors.iloc[:, : len(vectors.columns) - 1],
        value_vectors=pd.DataFrame(vectors['value'], columns=['value'])
    )
    return result