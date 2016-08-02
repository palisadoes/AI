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

    def __init__(self, data, bins=25):
        """Function for intializing the class.

        Args:
            data: List of tuples of format
                (class, feature_01, feature_02)

        """
        # Initialize key variables
        self.hgram = {}
        self.minmax = defaultdict(lambda: defaultdict(dict))
        values_by_class = defaultdict(lambda: defaultdict(dict))
        self.x_y = defaultdict(lambda: defaultdict(dict))

        # Calculate the number of bins using sturges
        # self.bin_count = int(math.log2(len(data)) + 1)
        self.bin_count = bins

        # Create a row for each column of data for each class (Transpose)
        for item in data:
            cls = item[0]
            values = item[1]

            # Track column values
            if bool(values_by_class[cls]) is False:
                values_by_class[cls] = [values]
            else:
                values_by_class[cls].append(values)

            # Get min / max values
            for column in range(0, len(values)):
                value = values[column]
                if bool(self.minmax[column]) is False:
                    self.minmax[column]['min'] = value
                    self.minmax[column]['max'] = value
                else:
                    self.minmax[column]['min'] = min(
                        value, self.minmax[column]['min'])
                    self.minmax[column]['max'] = max(
                        value, self.minmax[column]['max'])

                if bool(self.x_y[cls][column]) is False:
                    self.x_y[cls][column] = [value]
                else:
                    self.x_y[cls][column].append(value)

        print('Histogram')
        for column in range(0, len(values)):
            print(
                ('P%s') % (column + 1),
                self.minmax[column]['min'], self.minmax[column]['max'])

        # Create empty 2D array
        for cls in values_by_class.keys():
            self.hgram[cls] = np.zeros(
                (self.bin_count, self.bin_count))

        # Get bins data should be placed in
        for cls, tuple_list in sorted(values_by_class.items()):
            for values in tuple_list:
                (row, col) = self.row_col(values)

                # Update histogram
                self.hgram[cls][row][col] += 1

        # Create a list of classes found
        self.classes = sorted(values_by_class.keys())

    def row_col(self, dimensions):
        """Get the row and column for 2D histogram.

        Args:
            dimensions: Dimensions for histogram row / column allocation

        Returns:
            (row, col): Tuple of Row / Column for histogram

        """
        # Initialize key variables
        multiplier = self.bin_count - 1
        row_col = []

        # Calculate the row and column
        for idx, value in enumerate(dimensions):
            numerator = value - self.minmax[idx]['min']
            denominator = self.minmax[idx]['max'] - self.minmax[idx]['min']
            ratio = numerator / denominator
            row_col.append(
                # int(round(multiplier * ratio))
                int(multiplier * ratio)
            )

        # Return
        (row, col) = tuple(row_col)
        return (row, col)

    def classifier(self, dimensions):
        """Get the number of bins to use.

        Args:
            dimensions: Tuple of dimensions

        Returns:
            selection: Class classifier chooses

        """
        # Initialize key variables
        probability = {}

        # Get row / column for histogram for dimensions
        row, col = self.row_col(dimensions)

        if row >= self.bin_count or col >= self.bin_count:
            print(row, col)
            return None

        # Get the denominator
        denominator = self.hgram[self.classes[0]][row][col] + self.hgram[
            self.classes[1]][row][col]

        # Get probability of each class
        for cls in self.classes:
            # Do floating point math as numpy somtimes gives
            # "RuntimeWarning: invalid value encountered in double_scalars"
            # when dividing by very small numbers
            nominator = self.hgram[cls][row][col]
            if denominator == 0:
                probability[cls] = None
            else:
                probability[cls] = float(nominator) / float(denominator)

        # Reassign variables for readability
        prob_c0 = probability[self.classes[0]]
        prob_c1 = probability[self.classes[1]]

        # Evaluate probabilities
        if prob_c0 is None or prob_c1 is None:
            selection = None
        else:
            if prob_c0 + prob_c1 == 0:
                selection = None
            elif prob_c0 > prob_c1:
                selection = self.classes[0]
            elif prob_c0 < prob_c1:
                selection = self.classes[1]
            else:
                selection = None

        # Return
        return selection

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

    def graph3d(self):
        """Graph histogram.

        Args:

        Returns:
            None

        """
        # Initialize key variables
        directory = '/home/peter/Downloads'
        handles = []
        labels = []

        # Random colors for each plot
        prop_iter = iter(plt.rcParams['axes.prop_cycle'])

        # Initialize the figure
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')

        # Loop through data
        for cls in sorted(self.x_y.keys()):
            # Initialize various arrays
            x_positions = []
            y_positions = []
            z_positions = []
            z_height = []

            # Define data
            data = self.histogram()[cls]

            # Assign values to array
            for (x_pos, y_pos), z_pos in np.ndenumerate(data):
                # Setup lists to only plot when there are
                # meaningful values.
                if x_pos and y_pos and z_pos:
                    # Get coordinates for the bottom of the
                    # bar chart
                    x_positions.append(self._fixed_value(x_pos, 0))
                    y_positions.append(self._fixed_value(y_pos, 1))
                    z_positions.append(0)

                    # Keep track of the desired column height
                    z_height.append(z_pos)

            # Create elements defining the sides of each column
            num_elements = len(x_positions)
            x_col_width = np.ones(num_elements) * int(self.width(0))
            y_col_depth = np.ones(num_elements) * int(self.width(1))
            z_col_height = np.asarray(z_height)

            # Get color of plot
            color = next(prop_iter)['color']

            # Do the plot
            axes.bar3d(
                x_positions, y_positions, z_positions,
                x_col_width, y_col_depth, z_col_height,
                zsort='average',
                alpha=0.6,
                color=color)

            # Prepare values for legend
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=color))
            labels.append(str(cls).capitalize())

        # Add Main Title
        fig.suptitle(
            ('Class %s and Class %s Histogram (%s Bins)') % (
                str(self.classes[0]).capitalize(),
                str(self.classes[1]).capitalize(),
                self.bins()),
            horizontalalignment='center',
            fontsize=10)

        # Add legend
        axes.legend(handles, labels)

        # Add grid, axis labels
        axes.grid(True)
        axes.set_ylabel('Y Label')
        axes.set_xlabel('X Label')
        axes.set_zlabel('Count')

        # Adjust bottom
        fig.subplots_adjust(left=0.2, bottom=0.2)

        # Create image
        graph_filename = ('%s/homework-3-3D.png') % (directory)

        # Save chart
        fig.savefig(graph_filename)

        # Close the plot
        plt.close(fig)

    def _fixed_value(self, value, pointer):
        """Fix the value plotted on the histogram based on the bin.

        Args:
            value: Bin value
            pointer:

        Returns:

            fixed: Fixed value

        """
        # Initialize key variables
        width = self.width(pointer)

        # Calculate
        fixed = (value * width) + self.minmax[pointer]['min']

        # Return
        return fixed

    def width(self, pointer):
        """Fix the value plotted on the histogram based on the bin.

        Args:
            value: Bin value
            pointer:

        Returns:

            width: Fixed value

        """
        # Initialize key variables
        minimum = self.minmax[pointer]['min']
        maximum = self.minmax[pointer]['max']
        delta = maximum - minimum

        # Calculate
        width = delta / self.bins()

        # Return
        return width
