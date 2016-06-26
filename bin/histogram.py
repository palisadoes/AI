#!/usr/bin/env python3
"""Program creates histograms."""

import argparse
import textwrap
import math
import sys
from pprint import pprint
from collections import defaultdict
from statistics import stdev, mean, median

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

import xlrd


class Cli(object):
    """Class gathers all CLI information.

    Args:
        None

    Returns:
        None

    Functions:
        __init__:
        get_cli:
    """

    def __init__(self, additional_help=None, width=80):
        """Function for intializing the class."""
        # Initialize key variables
        self.width = width

        # Create a number of here-doc entries
        if additional_help is not None:
            self.config_help = additional_help
        else:
            self.config_help = ''

    def args(self):
        """Return all the CLI options.

        Args:
            self:

        Returns:
            args: Namespace() containing all of our CLI arguments as objects
                - filename: Path to the configuration file

        """
        # Header for the help menu of the application
        parser = argparse.ArgumentParser(
            description=self.config_help,
            formatter_class=argparse.RawTextHelpFormatter)

        # Add subparser
        subparsers = parser.add_subparsers(dest='mode')

        # Parse "histogram", return object used for parser
        self._histogram(subparsers)

        # Return the CLI arguments
        args = parser.parse_args()

        # Return our parsed CLI arguments
        return args

    def _histogram(self, subparsers):
        """Process "histogram" CLI commands.

        Args:
            subparsers: Subparsers object

        Returns:
            None

        """
        # Initialize key variables
        parser = subparsers.add_parser(
            'histogram',
            help=textwrap.fill(
                'Create histogram charts.', width=self.width)
        )

        # Process config_file
        parser.add_argument(
            '--datafile',
            required=True,
            default=None,
            type=str,
            help=textwrap.fill(
                'The data file to use.', width=self.width)
        )


class Data(object):
    """Class gathers all CLI information.

    Args:
        None

    Returns:
        None

    Functions:
        __init__:
        get_cli:
    """

    def __init__(self, filename):
        """Function for intializing the class."""
        # Initialize key variables
        self.data = []

        # Read spreadsheet
        workbook = xlrd.open_workbook(filename)
        worksheet = workbook.sheet_by_index(0)
        for row in range(worksheet.nrows):
            # Get data
            feet = worksheet.row(row)[0].value
            inches = worksheet.row(row)[1].value
            gender = worksheet.row(row)[2].value

            # Skip header, append data
            if 'gender' not in gender.lower():
                self.data.append(
                    (int(feet * 12 + inches), gender.lower()))

    def keys(self):
        """Get keys in data.

        Args:
            None

        Returns:
            data: List of keys

        """
        # Initialize key variables
        data = []

        # Get keys
        for pair in self.data:
            key = pair[1]
            if key in keys:
                continue
            data.append(key)

        # Return
        return data

    def buckets(self, category):
        """Get number of buckets in data.

        Args:
            None

        Returns:
            data: number of buckets

        """
        # Initialize key variables
        lists = self.lists()
        data = int(math.log2(len(lists[category])) + 1)

        # Return
        return data

    def counts(self):
        """Convert file data to probabilty counts.

        Args:
            None

        Returns:
            counts: list of tuples (height, gender)

        """
        # Initialize key variables
        keys = []
        counts = defaultdict(lambda: defaultdict(dict))

        # Create counts dict for probabilities
        for pair in self.data:
            height = pair[0]
            gender = pair[1]
            if gender in counts:
                if height in counts[gender]:
                    counts[gender][height] = counts[
                        gender][height] + 1
                else:
                    counts[gender][height] = 1
            else:
                counts[gender][height] = 1

        # Return
        return counts

    def lists(self):
        """Convert count data lists.

        Args:
            None

        Returns:
            data: Dict of histogram data keyed by category (gender)

        """
        # Initialize key variables
        frequencies = self.counts()
        data = {}

        # Create counts dict for probabilities
        for pair in self.data:
            height = pair[0]
            gender = pair[1]
            if gender in data:
                data[gender].append(height)
            else:
                data[gender] = [height]

        # Return
        return data


    def probability(self, category=None, height=None):
        """Calculate probabilities.

        Args:
            category: Category of data
            height: Height to calculate probaility for

        Returns:
            value: Value of probability

        """
        # Initialize key variables
        counts = self.counts()
        total = 0
        frequency = 0

        if category is None:
            for gender in counts.keys():
                for key in counts[gender].keys():
                    total = total + counts[gender][key]
                frequency = counts[gender][height]
        else:
            for gender in counts.keys():
                for gender_height in counts[gender].keys():
                    if category == gender:
                        total = total + counts[gender][gender_height]
            frequency = counts[category][height]

        # Return
        value = frequency / total
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
        data = self.lists()

        # Loop through data
        for category in data:
            # Create objects
            bins = self.buckets(category)
            histogram_array = np.array(data[category])
            histogram, bins = np.histogram(histogram_array, bins=bins)

            # Calculate widths and center line of histogram
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2

            # Create the histogram plot
            fig, axes = plt.subplots(
            figsize=(8, 8))
            axes.bar(
                center, histogram, align='center', width=width, color='#008db8')

            # create the median plot
            handle_median = axes.axvline(
                x=median(histogram_array),
                linewidth=4, color='#FFD700', alpha=.6
            )

            # create the mean plot
            handle_mean = axes.axvline(
                x=mean(histogram_array),
                linewidth=4, color='#000000', alpha=.6
            )

            # Add legend
            fig.legend(
                (handle_median, handle_mean), ('Median', 'Mean'),
                loc='lower center', ncol=2)

            # Add Main Title
            fig.suptitle(('%s Histogram') % (
                category.capitalize()), fontsize=10)

            # Add Subtitle
            axes.set_title(
                ('Mean: %.5f, Median: %.5f StdDev: %.5f') % (
                     mean(histogram_array),
                     median(histogram_array),
                     stdev(histogram_array)),
                fontsize=8)

            # Add grid, axis labels
            axes.grid(True)
            axes.set_ylabel('Count')
            axes.set_xlabel(category.capitalize())

            # Rotate the labels
            for label in axes.xaxis.get_ticklabels():
                label.set_rotation(90)

            # Adjust bottom
            fig.subplots_adjust(left=0.2, bottom=0.2)

            # Create image
            graph_filename = ('%s/%s.png') % (directory, category)

            # Save chart
            fig.savefig(graph_filename)

            # Close the plot
            plt.close(fig)


def main():
    """Process data."""
    # Process the CLI
    cli = Cli()
    args = cli.args()

    # We are only doing histogram stuff
    if args.mode != 'histogram':
        sys.exit(0)

    # Get data
    data = Data(args.datafile)
    data.graph()
    print(data.probability(category='male', height=65))
    print(data.probability(height=65))


if __name__ == "__main__":
    main()
