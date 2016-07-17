#!/usr/bin/env python3
"""Program creates histograms."""

# Standard python imports
import sys
import math
from collections import defaultdict
from pprint import pprint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Non-standard python imports
import numpy as np


class PCA(object):
    """Class for 2 dimensional histogram.

    Args:
        None

    Returns:
        None

    Functions:
        __init__:
        get_cli:
    """

    def __init__(self, data):
        """Function for intializing the class.

        Args:
            data: List of tuples of format
                (class, dimension1, dimension2 ...)

        """
        # Initialize key variables
        self.data = data
        classes = {}
        class_rows = {}
        self.x_values = {}
        minmax = defaultdict(lambda: defaultdict(dict))
        values_by_class = defaultdict(lambda: defaultdict(dict))
        self.x_y = defaultdict(lambda: defaultdict(dict))

        # Determine the number of dimensions in vector
        for _, vector in data:
            dimensions = len(vector)
            break

        for cls, vector in data:
            classes[cls] = None
            if cls in class_rows:
                class_rows[cls].append(0)
            else:
                class_rows[cls] = [0]

        # Create a numpy array for the class
        for cls in classes.keys():
            self.x_values[cls] = np.zeros(
                len(class_rows[cls]), dimensions)

        # Create a numpy array for the class


        # Create a row for each column of data for each class (Transpose)
        for cls, vectors in data:
            cls = item[0]
            vectors = item[1]

            # Track column values
            if bool(values_by_class[cls]) is False:
                values_by_class[cls] = [values]
            else:
                values_by_class[cls].append(values)

            # Get min / max values
            for column in range(0, len(values)):
                value = values[column]
                if bool(minmax[column]) is False:
                    minmax[column]['min'] = value
                    minmax[column]['max'] = value
                else:
                    minmax[column]['min'] = min(value, minmax[column]['min'])
                    minmax[column]['max'] = max(value, minmax[column]['max'])

                if bool(self.x_y[cls][column]) is False:
                    self.x_y[cls][column] = [value]
                else:
                    self.x_y[cls][column].append(value)

        # Create empty 2D array
        for cls in values_by_class.keys():
            self.x_values[cls] = np.zeros(
                (self.bin_count, self.bin_count))

        # Get bins data should be placed in
        for cls, tuple_list in values_by_class.items():
            for values in tuple_list:
                row = self._placement(values[0], minmax[0])
                col = self._placement(values[1], minmax[1])

                # Update histogram
                self.x_values[cls][row][col] += 1

        # Assign global variables
        self.minmax = minmax

    def _placement(self, value, minmax):
        """Get the row or column for 2D histogram.

        Args:
            value: Value to classify
            minmax: Dict of minimum / maximum to use

        Returns:
            hbin: Row / Column for histogram

        """
        # Initialize key variables
        multiplier = self.bin_count - 1

        # Calculate
        maximum = minmax['max']
        minimum = minmax['min']
        ratio = (value - minimum) / (maximum - minimum)
        hbin = int(round(multiplier * ratio))

        # Return
        return hbin

