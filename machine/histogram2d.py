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

    def __init__(self, data, labels):
        """Function for intializing the class.

        Args:
            data: List of tuples of format
                (class, dimension1, dimension2 ...)
            labels: Labels for data columns

        """
        # Initialize key variables
        self.data = data
        self.labels = labels

        self.hgram = {}
        minmax = defaultdict(lambda: defaultdict(dict))
        values_by_class = defaultdict(lambda: defaultdict(dict))
        self.x_y = defaultdict(lambda: defaultdict(dict))

        # Calculate the number of bins using sturges
        self.bin_count = int(math.log2(len(data)) + 1)
        self.bin_count = 15

        # Create a row for each column of data for each class (Transpose)
        for item in data:
            cls = item[0]
            values = (item[1], item[2])

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
            self.hgram[cls] = np.zeros(
                (self.bin_count, self.bin_count))

        # Get bins data should be placed in
        for cls, tuple_list in values_by_class.items():
            for values in tuple_list:
                row = self._placement(values[0], minmax[0])
                col = self._placement(values[1], minmax[1])

                # Update histogram
                self.hgram[cls][row][col] += 1

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

    def graph2d(self):
        """Graph histogram.

        Args:
            histogram_list: List for histogram
            cls: Category (label) for data in list
            bins: Number of bins to use

        Returns:
            None

        """
        # Initialize key variables
        directory = '/home/peter/Downloads'
        nbins = self.bins()

        # Loop through data
        for cls in sorted(self.x_y.keys()):
            # Get key data for creating histogram
            x_array = np.array(self.x_y[cls][0])
            y_array = np.array(self.x_y[cls][1])

            # Estimate the 2D histogram
            hgram, xedges, yedges = np.histogram2d(
                x_array, y_array, bins=nbins)

            # hgram needs to be rotated and flipped
            hgram = np.rot90(hgram)
            hgram = np.flipud(hgram)

            # Mask zeros
            # Mask pixels with a value of zero
            hgram_masked = np.ma.masked_where(hgram == 0, hgram)

            # Plot 2D histogram using pcolor
            fig = plt.figure()
            plt.pcolormesh(xedges, yedges, hgram_masked)
            plt.xlabel('Height')
            plt.ylabel('Handspan')
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Counts')

            # Add Main Title
            fig.suptitle(
                ('Height and Handspan Histogram (%ss, %s Bins)') % (
                    cls.capitalize(), nbins),
                horizontalalignment='center',
                fontsize=10)

            # Create image
            graph_filename = (
                '%s/homework-2-2D-%s-bins-%s.png'
                '') % (directory, cls, nbins)

            # Save chart
            fig.savefig(graph_filename)

            # Close the plot
            plt.close(fig)

    def graph3d(self):
        """Graph histogram.

        Args:

        Returns:
            None

        """
        # Initialize key variables
        directory = '/home/peter/Downloads'
        bins = self.bins()
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
            x_col_width = np.ones(num_elements)
            y_col_depth = np.ones(num_elements)
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
            labels.append(cls.capitalize())

        # Add Main Title
        fig.suptitle(
            ('%s and %s Histogram (%s Bins)') % (
                self.labels[0].capitalize(),
                self.labels[1].capitalize(),
                self.bins()),
            horizontalalignment='center',
            fontsize=10)

        # Add legend
        axes.legend(handles, labels)

        # Add grid, axis labels
        axes.grid(True)
        axes.set_ylabel('Handspans')
        axes.set_xlabel('Heights')
        axes.set_zlabel('Count')

        # Adjust bottom
        fig.subplots_adjust(left=0.2, bottom=0.2)

        # Create image
        graph_filename = ('%s/homework-2-3D.png') % (directory)

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
        minimum = self.minmax[pointer]['min']
        maximum = self.minmax[pointer]['max']
        delta = maximum - minimum

        # Calculate
        fixed = (value * delta / self.bins()) + self.minmax[pointer]['min']

        # Return
        return fixed

    def _width_value(self, pointer):
        """Fix the width of the value plotted on the histogram based on the bin.

        Args:
            value: Bin value
            pointer:

        Returns:

            fixed: Fixed value

        """
        # Initialize key variables
        minimum = self.minmax[pointer]['min']
        maximum = self.minmax[pointer]['max']
        delta = maximum - minimum

        # Calculate
        fixed = self.bins() / delta

        # Return
        return fixed
