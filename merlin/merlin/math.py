"""Library to process the ingest of data files."""

# PIP imports
import pandas as pd


class Difference(object):
    """Convert Pandas DataFrame to components."""

    def __init__(self, data):
        """Initialize the class.

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
        """Initialize the class.

        Args:
            data: Pandas DataFrame with columns:
                ['open', 'high', 'low', 'close', 'volume']
            window: Size of rolling window

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
            None

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
            window: Size of rolling window

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


class Misc(object):
    """Convert Pandas DataFrame to components."""

    def __init__(self, data):
        """Initialize the class.

        Args:
            data: Pandas DataFrame with columns:
                ['open', 'high', 'low', 'close', 'volume']

        Returns:
            None

        """
        # Initialize key variables
        self._data = data

    def proc(self, window):
        """Calculate the PROC (Price Rate of Change) within N days.

        Calculated based on the formula at:
        https://en.wikipedia.org/wiki/Relative_strength_index

        Args:
            window: Size of rolling window

        Returns:
            result: Pandas Series

        """
        # Initialize key variables
        close = self._data['close'].values
        result = [0] * len(close)

        # Get a rolling list of values of size window from closing values
        _ranges = [
            close[i: i + window] for i in range(len(close) - (window - 1))]

        # Create a list of the percentage deltas of the start and ending
        # values in each rolling window
        for index in range(window, len(close)):
            first = _ranges[index - window][0]
            last = _ranges[index - window][window - 1]
            result[index] = (last - first) / last

        # print('\n\n', _ranges, '\n\n', result, '\n\n')
        # sys.exit(0)

        # Return
        return result

    def rsi(self, window):
        """Calculate the RSI (Relative Strength Index) within N days.

        Calculated based on the formula at:
        https://en.wikipedia.org/wiki/Relative_strength_index

        Args:
            window: Size of rolling window

        Returns:
            result: Pandas Series

        """
        # Initialize key variables
        difference = Difference(self._data)
        close = difference.actual()['close']

        # Track positive and negative deltas
        dataframe = pd.DataFrame(columns=['positive', 'negative'])
        dataframe['positive'] = (close + close.abs()) / 2
        dataframe['negative'] = (-close + close.abs()) / 2

        # Calculate the smoothed simple moving average
        positive_ema = smma(dataframe['positive'], window)
        negative_ema = smma(dataframe['negative'], window)

        # Return
        relative_strength = positive_ema / negative_ema
        result = 100 - (100 / (1.0 + relative_strength))
        return result


def smma(data, window):
    """Calculate the smooth modified moving average.

    Args:
        data: Pandas Series
        window: Moving average window

    Returns:
        result: Pandas Series

    """
    # Return
    result = data.ewm(
        ignore_na=False, alpha=1.0 / window,
        min_periods=0, adjust=True).mean()
    return result
