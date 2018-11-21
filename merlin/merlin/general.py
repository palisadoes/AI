"""Library to process the ingest of data files."""

# PIP imports
import pandas as pd


class Dates(object):
    """Convert Pandas date series to components."""

    def __init__(self, dates, date_format):
        """Function for intializing the class.

        Args:
            dates: Pandas series of dates
            date_format: Date format

        Returns:
            None

        """
        # Get date values from data
        self.weekday = pd.to_datetime(dates, format=date_format).dt.weekday
        self.day = pd.to_datetime(dates, format=date_format).dt.day
        self.dayofyear = pd.to_datetime(
            dates, format=date_format).dt.dayofyear
        self.quarter = pd.to_datetime(dates, format=date_format).dt.quarter
        self.month = pd.to_datetime(dates, format=date_format).dt.month
        self.week = pd.to_datetime(dates, format=date_format).dt.week
        self.year = pd.to_datetime(dates, format=date_format).dt.year
