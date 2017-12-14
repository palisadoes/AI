#!/usr/bin/env python3
"""Script to forecast data using CNN AI."""

# Standard imports
import sys
import argparse
import csv
from pprint import pprint
import time

# Pip imports
import numpy as np


class Data(object):
    """Process data for ingestion."""

    def __init__(self, filename, ts_start=None):
        """Method that instantiates the class.

        Args:
            filename: Name of CSV file to read
            ts_start: Starting timestamp for which data should be retrieved

        Returns:
            None

        """
        # Initialize key variables
        self._data = {}
        rrd_step = 300
        now = _normalize(int(time.time()), rrd_step)

        # Set the start time to be 2 years by default
        if (ts_start is None) or (ts_start < 0):
            # ts_start = now - (3600 * 24 * 365 * 2)
            ts_start = now - (3600 * 24 * 30)

        # Set the stop time to be now by default
        ts_stop = now

        # Pre-populate the data dictionary with zeros
        # (just in case there is missing data in the data source)
        timestamps = range(
            _normalize(ts_start, rrd_step),
            _normalize(ts_stop, rrd_step),
            rrd_step)
        for timestamp in timestamps:
            self._data[timestamp] = 0

        # Read data
        with open(filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                timestamp = _normalize(int(row[0]), rrd_step)
                if ts_start <= timestamp <= ts_stop:
                    value = float(row[1])
                    self._data[timestamp] = value

        self._ts_start = ts_start
        self._ts_stop = ts_stop

    def sample(self, hours=1):
        """Create histogram of data .

        Args:
            filename: Name of CSV file to read
            ts_start: Starting timestamp for which data should be retrieved
            ts_stop: Stopping timestamp for which data should be retrieved

        Returns:
            None

        """
        # Initialize key variables
        interval = hours * 3600
        data_dict = {}
        keys = list(sorted(self._data.keys()))
        buckets = list(range(
            _normalize(self._ts_start, interval),
            _normalize(self._ts_stop, interval),
            interval))

        pointer = 0
        print('Buckets:', min(buckets), max(buckets), len(buckets))
        print('Start / Stop:', self._ts_start, self._ts_stop)
        print('Keys', min(keys), max(keys))

        # Get the start and stop boundaries of each bucket
        boundaries = []
        for pointer in range(1, len(buckets)):
            boundaries.append(
                (buckets[pointer], buckets[pointer - 1])
            )

        # Sum the values within the boundaries
        for boundary in boundaries:
            ts_start = boundary[0]
            ts_stop = boundary[1]
            values = [0]
            print(ts_start, ts_stop)
            print(self._data[ts_start], self._data[ts_stop])
            for timestamp in range(ts_start, ts_stop):
                values.append(self._data[timestamp])
            data_dict[ts_stop] = max(values)
            sys.exit(0)
            # print(ts_stop, max(values))

        # pprint(data_dict)


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


def main():
    """Main Function.

    Display data prediction from tensorflow model

    """
    # Get filename
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='Name of CSV file to read.',
        type=str, required=True)
    args = parser.parse_args()
    filename = args.filename

    # Open file and get data
    data = Data(filename)
    data.sample()


if __name__ == "__main__":
    main()
