"""Library to process the ingest of data files."""

# PIP imports
import pandas as pd


class Difference(object):
    """Convert Pandas DataFrame to components."""

    def __init__(self, data):
        """Function for intializing the class.

        Args:
            data: Pandas DataFrame with columns:
                ['open', 'high', 'low', 'close', 'volume']

        Returns:
            None

        """
        # Initialize key variables
        self._data = data

    def relative(self):
        """Find the percentage difference between successive DataFrame rows.

        Args:
            percent: Give percentage differences if True otherwise return
                numeric differences.

        Returns:
            result: DataFrame with columns:
                ['open', 'high', 'low', 'close', 'volume']

        """
        # Calculate the percentage differences between columns
        result = self._data.pct_change()

        # Return
        return result

    def actual(self):
        """Find the actual difference between successive DataFrame rows.

        Args:
            None

        Returns:
            result: DataFrame with columns:
                ['open', 'high', 'low', 'close', 'volume']

        """
        # Calculate the percentage and real differences between columns
        result = self._data.diff()

        # Return
        return result


class Stochastic(object):
    """Convert Pandas DataFrame to components."""

    def __init__(self, data, window=35):
        """Function for intializing the class.

        Args:
            data: Pandas DataFrame with columns:
                ['open', 'high', 'low', 'close', 'volume']
            window: Rolling window

        Returns:
            None

        """
        # Initialize key variables
        self._data = data
        _window = int(abs(window))

        # Create the "RollingLow" column in the DataFrame
        self._rolling_low = self._data['low'].rolling(window=_window).min()

        # Create the "RollingHigh" column in the DataFrame
        self._rolling_high = self._data['high'].rolling(window=_window).max()

    def k(self):
        """Calculate Stochastic percentage K values.

        Args:
            Mpme

        Returns:
            result: Series of Stochastic K percentage values
                ['open', 'high', 'low', 'close', 'volume']

        """
        # Create the "%K" column in the DataFrame
        result = 100 * (
            (self._data['close'] - self._rolling_low) / (
                self._rolling_high - self._rolling_low))

        # Return
        return result

    def d(self, window=5):
        """Calculate Stochastic percentage D values.

        Args:
            percent: Give percentage differences if True

        Returns:
            result: DataFrame with columns:
                ['open', 'high', 'low', 'close', 'volume']

        """
        # Initialize key variables
        _window = int(abs(window))

        # Create the "%D" column in the DataFrame
        result = self.k().rolling(window=_window).mean()

        # Return
        return result
