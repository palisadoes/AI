"""Library to process the ingest of data files."""

# Standard imports
import time
import csv
import os

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


def save_trials(trials, input_filename):
    """Save trial results to file.

    Args:
        trials: List of trial results dicts

    Returns:
        None

    """
    # Initialize key variables
    filename = (
        '/tmp/trials-{}-{}.csv'.format(
            os.path.basename(input_filename), int(time.time())))
    count = 0

    # Iterate
    for trial in trials:
        # Identify variables to output
        hyperparameters = trial['result']['hyperparameters']
        hyperparameters['loss'] = trial['result']['loss']
        hyperparameters['index'] = int(trials['misc']['tid'])

        with open(filename, 'a', newline='') as csvfile:
            fieldnames = list(sorted(hyperparameters.keys()))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Create the header
            if count == 0:
                writer.writeheader()
                count += 1

            # Write data
            writer.writerow(hyperparameters)
