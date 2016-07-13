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
        bins = self.bins()
        categories = []
        lines2plot = []

        # Create the histogram plot
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')

        # Random colors for each plot
        prop_iter = iter(plt.rcParams['axes.prop_cycle'])

        # Loop through data
        for category in self.x_y.keys():
            # Get key data for creating histogram
            x_array = np.array(self.x_y[category][0])
            y_array = np.array(self.x_y[category][1])
            (hist, xedges, yedges) = np.histogram2d(
                x_array, y_array, bins=bins)

            # Number of boxes
            elements = (len(xedges) - 1) * (len(yedges) - 1)
            (xpos, ypos) = np.meshgrid(
                xedges[:-1] + 0.25, yedges[:-1] + 0.25)

            # x and y coordinates of the bars
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros(elements)

            # Lengths of the bars on relevant axes
            dx_length = 1.0 * np.ones_like(zpos)
            dy_length = dx_length.copy()
            dz_length = hist.flatten()

            # Append category name
            categories.append(category.capitalize())

            # Chart line
            lines2plot = axes.bar3d(
                xpos, ypos, zpos,
                dx_length, dy_length, dz_length,
                alpha=0.5,
                zsort='average',
                color=next(prop_iter)['color'],
                label=category.capitalize())


        """

        # Put ticks only on bottom and left
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('bottom')
        axes.zaxis.set_ticks_position('bottom')

        # Set X axis ticks
        major_ticks = np.arange(0, bins, 1)
        axes.set_xticks(major_ticks)

        # Set y axis ticks
        major_ticks = np.arange(0, bins, 1)
        axes.set_yticks(major_ticks)

        # Set z axis ticks
        major_ticks = np.arange(0, max(dz_length), 5)
        axes.set_zticks(major_ticks)

        """

        # Add legend
        # axes.legend(lines, categories)
        # plt.legend()

        # Add Main Title
        fig.suptitle(
            'Height and Handspan Histogram',
            horizontalalignment='center',
            fontsize=10)

        # Add grid, axis labels
        axes.grid(True)
        axes.set_ylabel('Handspans')
        axes.set_xlabel('Heights')
        axes.set_zlabel('Count')

        # Adjust bottom
        fig.subplots_adjust(left=0.2, bottom=0.2)

        # Create image
        graph_filename = ('%s/homework-2.png') % (directory)

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
