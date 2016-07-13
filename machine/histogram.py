#!/usr/bin/env python3
"""Program creates histograms."""

# Standard python imports
import sys
import math
from collections import defaultdict
from pprint import pprint

import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Non-standard python imports
import numpy as np


class Histogram2D(object):
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
        self.hgram = {}
        column_by_class = defaultdict(lambda: defaultdict(dict))
        column_no_class = defaultdict(lambda: defaultdict(dict))

        # Calculate the number of bins using sturges
        self.bin_count = int(math.log2(len(data)) + 1)

        # Create a row for each column of data for each class (Transpose)
        for item in data:
            cls = item[0]
            values = (item[1], item[2])

            # Get values for each class
            for column in range(0, len(values)):
                # Get column value
                value = values[column]

                # Track column values
                if bool(column_by_class[cls][column]) is False:
                    column_by_class[cls][column] = [value]
                else:
                    column_by_class[cls][column].append(value)

                # Track values without classes
                if bool(column_no_class[column]) is False:
                    column_no_class[column] = [value]
                else:
                    column_no_class[column].append(value)

        # Create empty 2D array
        for cls in column_by_class.keys():
            self.hgram[cls] = np.zeros(
                (self.bin_count, self.bin_count))

        # Get bins data should be placed in
        for cls in column_by_class.keys():
            for row in range(0, len(column_by_class[cls][0])):
                # Get bins data should be placed in
                v4row = column_by_class[cls][0][row]
                hrow = _row_col(v4row, column_no_class[0], self.bin_count)

                v4col = column_by_class[cls][1][row]
                hcol = _row_col(v4col, column_no_class[1], self.bin_count)

                # Update histogram
                self.hgram[cls][hrow][hcol] += 1

        #
        self.x_y = column_by_class

    def bins(self):
        """Get the number of bins to use.

        Args:
            None

        Returns:
            value: number of bins to use

        """
        # Return
        value = self.bin_count
        return value

    def histogram(self):
        """Get the histogram.

        Args:
            None

        Returns:
            value: 2D histogram

        """
        value = self.hgram
        return value

    def graph(self):
        """Graph histogram.

        Args:
            histogram_list: List for histogram
            category: Category (label) for data in list
            bins: Number of bins to use

        Returns:
            None

        """
        # Initialize key variables
        directory = '/home/peter/Downloads'
        nbins = self.bins()
        # nbins = 15

        # Loop through data
        for category in self.x_y.keys():
            # Get key data for creating histogram
            x_array = np.array(self.x_y[category][0])
            y_array = np.array(self.x_y[category][1])

            # Estimate the 2D histogram
            hgram, xedges, yedges = np.histogram2d(
                x_array, y_array, bins=nbins)

            # hgram needs to be rotated and flipped
            hgram = np.rot90(hgram)
            hgram = np.flipud(hgram)

            # Mask zeros
            # Mask pixels with a value of zero
            hgram_masked = np.ma.masked_where(hgram==0, hgram)

            # Plot 2D histogram using pcolor
            fig = plt.figure()
            plt.pcolormesh(xedges,yedges,hgram_masked)
            plt.xlabel('Height')
            plt.ylabel('Handspan')
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Counts')

            # Add Main Title
            fig.suptitle(
                ('Height and Handspan Histogram (%ss, %s Bins)') % (
                    category.capitalize(), nbins),
                horizontalalignment='center',
                fontsize=10)

            # Create image
            graph_filename = (
                '%s/homework-2-%s-bins-%s.png'
                '') % (directory, category, nbins)

            # Save chart
            fig.savefig(graph_filename)

            # Close the plot
            plt.close(fig)


def _row_col(value, column_data, bin_count):
    """Get the row or column for 2D histogram.

    Args:
        value: Value to search for
        column_data: Classes of data
        bin_count: Number of bins

    Returns:
        hbin: Row / Column for histogram

    """
    # Initialize key variables
    multiplier = bin_count - 1

    # Calculate
    maximum = max(column_data)
    minimum = min(column_data)
    ratio = (value - minimum) / (maximum - minimum)
    hbin = int((multiplier * ratio))

    # Return
    return hbin
