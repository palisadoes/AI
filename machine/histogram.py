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
            category: Category (label) for data in list
            bins: Number of bins to use

        Returns:
            None

        """
        # Initialize key variables
        directory = '/home/peter/Downloads'
        nbins = self.bins()

        # Loop through data
        for category in sorted(self.x_y.keys()):
            # Get key data for creating histogram
            x_array = np.array(self.x_y[category][0])
            y_array = np.array(self.x_y[category][1])

            # Estimate the 2D histogram
            hgram, xedges, yedges = np.histogram2d(
                x_array, y_array, bins=nbins)

            print(category)
            pprint(hgram)

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
                    category.capitalize(), nbins),
                horizontalalignment='center',
                fontsize=10)

            # Create image
            graph_filename = (
                '%s/homework-2-2D-%s-bins-%s.png'
                '') % (directory, category, nbins)

            # Save chart
            fig.savefig(graph_filename)

            # Close the plot
            plt.close(fig)

    def graph3d(self):
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

        # Random colors for each plot
        prop_iter = iter(plt.rcParams['axes.prop_cycle'])

        # Loop through data
        for category in sorted(self.x_y.keys()):
            # Create the histogram plot
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')

            # Get key data for creating histogram
            x_array = np.array(self.x_y[category][0])
            y_array = np.array(self.x_y[category][1])
            (hist, xedges, yedges) = np.histogram2d(
                x_array, y_array, bins=bins)

            print('-----------------------')
            print(category)
            pprint(hist)

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
            axes.bar3d(
                xpos, ypos, zpos,
                dx_length, dy_length, dz_length,
                alpha=0.5,
                zsort='average',
                color=next(prop_iter)['color'],
                label=category.capitalize())

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
            graph_filename = (
                '%s/homework-2-3D-%s-bins-%s.png'
                '') % (directory, category, bins)

            # Save chart
            fig.savefig(graph_filename)

            # Close the plot
            plt.close(fig)
