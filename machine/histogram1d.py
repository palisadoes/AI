#!/usr/bin/env python3
"""Program creates histograms."""

import argparse
import textwrap
import sys
import math
from collections import defaultdict
from statistics import stdev, mean

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

import xlrd


class Histogram1D(object):
    """Class gathers all CLI information.

    Args:
        None

    Returns:
        None

    Functions:
        __init__:
        get_cli:
    """

    def __init__(self, data, dimension):
        """Function for intializing the class.

        Args:
            data: List tuples [(value, class)]
            dimension: Category of data

        """
        # Initialize key variables
        self.data = data
        self.dimension = dimension
        self.meta = defaultdict(lambda: defaultdict(dict))

        # Calculate counts
        self.data_lists = self._lists()

        # Get min / max and bins
        for category, heights in sorted(self.data_lists.items()):
            self.meta[category]['min'] = min(heights)
            self.meta[category]['max'] = max(heights)

        # Calculate counts
        self.data_counts = self._counts()

        # Calculate probabilities
        self.data_probability = self._probability()

        # Get the number of entries
        self.entries = len(data)

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
            if key in data:
                continue
            data.append(key)

        # Return
        return data

    def bucket_range(self):
        """Get largest number of buckets out of all categories.

        Args:
            None

        Returns:
            (min, max): Tuple

        """
        # Initialize key variables
        for category in self.meta.keys():
            minimum = self.meta[category]['min']
            maximum = self.meta[category]['max']

        # Get range
        for category in self.meta.keys():
            minimum = min(self.meta[category]['min'], minimum)
            maximum = max(self.meta[category]['max'], maximum)

        # Return
        return (int(minimum), int(maximum))

    def buckets(self, category):
        """Get number of buckets in data.

        Args:
            category: Category of data

        Returns:
            bins: number of buckets

        """
        # Initialize key variables
        bins = self._buckets(category)[0]

        # Return
        return bins

    def minimum(self, category):
        """Get minimum category key value.

        Args:
            category: Category of data

        Returns:
            data: minimum value

        """
        # Initialize key variables
        data = self._buckets(category)[1]

        # Return
        return data

    def maximum(self, category):
        """Get maximum category key value.

        Args:
            category: Category of data

        Returns:
            data: maximum value

        """
        # Initialize key variables
        data = self._buckets(category)[2]

        # Return
        return data

    def counts(self):
        """Convert file data to probabilty counts.

        Args:
            None

        Returns:
            data: Dict of counts of height keyed by category and height

        """
        # Initialize key variables
        data = self.data_counts

        # Return
        return data

    def lists(self):
        """Convert count data lists.

        Args:
            None

        Returns:
            data: Dict of histogram data keyed by category (gender)

        """
        # Initialize key variables
        data = self.data_lists

        # Return
        return data

    def probability(self, height, category=None):
        """Calculate probabilities.

        Args:
            category: Category of data
            height: Height to calculate probaility for

        Returns:
            value: Value of probability

        """
        # Initialize key variables
        if category is None:
            value = self.data_probability[None][height]
        else:
            value = self.data_probability[category][height]

        # Return
        return value

    def table(self):
        """Create classifier chart.

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        (minimum, maximum) = self.bucket_range()

        # Header
        output = ('%10s %15s %15s %15s %15s %6s %6s') % (
            self.dimension.capitalize(),
            'Gender (H)', 'Prob. (H)',
            'Gender (B)', 'Prob. (B)',
            'Male', 'Female')
        print(output)

        # Evaluate heights
        for height in range(minimum, maximum + 1):
            # Get probabilities
            b_probability = self.bayesian_classifier(height)
            h_probability = self.histogram_classifier(height)

            # Get genders
            b_gender = _get_gender(b_probability)
            h_gender = _get_gender(h_probability)

            # Get counts
            mcount = self.data_counts['male'][height]
            if bool(mcount) is False:
                mcount = 0
            fcount = self.data_counts['female'][height]
            if bool(fcount) is False:
                fcount = 0

            # Account for zero males / females for height
            if mcount + fcount == 0:
                b_gender = 'N/A'
                h_gender = 'N/A'

            # Print output
            output = ('%10d %15s %15.6f %15s %15.6f %6d %6d') % (
                height,
                h_gender, h_probability,
                b_gender, b_probability,
                mcount, fcount)
            print(output)

    def bayesian_classifier(self, height):
        """Create histogram classifier chart.

        Args:
            None

        Returns:
            male_probability: Probability of being male

        """
        # Get male counts
        mcount = self._bayesian('male', height)

        # Get female counts
        fcount = self._bayesian('female', height)

        # Get male probability
        if mcount == 0:
            male_probability = 0
        else:
            male_probability = mcount / (mcount + fcount)

        # Return
        return male_probability

    def histogram_classifier(self, height):
        """Create histogram classifier chart.

        Args:
            None

        Returns:
            None

        """
        # Get male counts
        if height not in self.data_counts['male']:
            mcount = 0
        else:
            mcount = self.data_counts['male'][height]

        # Get female counts
        if height not in self.data_counts['female']:
            fcount = 0
        else:
            fcount = self.data_counts['female'][height]

        # Get male probability
        if mcount == 0:
            male_probability = 0
        else:
            male_probability = mcount / (mcount + fcount)

        # Return
        return male_probability

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
        dimension = self.dimension.capitalize()
        data = self.counts()
        categories = []
        heights = {}
        counts = {}
        min_height = 100000000000000000000
        max_height = 1 - min_height
        max_count = 1 - 100000000000000000000

        # Create the histogram plot
        fig, axes = plt.subplots(figsize=(7, 7))

        # Random colors for each plot
        prop_iter = iter(plt.rcParams['axes.prop_cycle'])

        # Loop through data
        for category in data:
            # Create empty list of heights
            heights[category] = []
            counts[category] = []

            # Append category name
            categories.append(category.capitalize())

            # Create lists to chart
            for height, count in sorted(data[category].items()):
                heights[category].append(height)
                counts[category].append(count)

            # Get max / min heights
            max_height = max(max_height, max(heights[category]))
            min_height = min(min_height, min(heights[category]))

            # Get max / min counts
            max_count = max(max_count, max(counts[category]))

            # Chart line
            plt.plot(
                heights[category], counts[category],
                color=next(prop_iter)['color'], label=category.capitalize())

        # Put ticks only on bottom and left
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('left')

        # Set X axis ticks
        major_ticks = np.arange(min_height, max_height, 1)
        axes.set_xticks(major_ticks)

        # Set y axis ticks
        major_ticks = np.arange(0, max_count + 100, 50)
        axes.set_yticks(major_ticks)

        # Add legend
        plt.legend()
        #fig.legend(
        #    tuple(lines), tuple(categories), loc='lower center', ncol=2)

        # Add Main Title
        fig.suptitle(
            ('%s Histogram') % (dimension),
            horizontalalignment='center',
            fontsize=10)

        # Add Subtitle
        axes.set_title(
            ('Analysis of First %s Rows') % (
                self.entries),
            multialignment='center',
            fontsize=10)

        # Add grid, axis labels
        axes.grid(True)
        axes.set_ylabel('Count')
        axes.set_xlabel(dimension)

        # Rotate the labels
        for label in axes.xaxis.get_ticklabels():
            label.set_rotation(90)

        # Adjust bottom
        fig.subplots_adjust(left=0.2, bottom=0.2)

        # Create image
        graph_filename = ('%s/homework-%s-%s-rows.png') % (
            directory, self.dimension, self.entries)

        # Save chart
        fig.savefig(graph_filename)

        # Close the plot
        plt.close(fig)

    def parameters(self):
        """Print gausian parameters for probability distribution function.

        Args:
            None

        Returns:
            None

        """
        # Print summary information
        output = ('%-25s: %s') % ('Total Sample Size', len(self.data))
        print(output)

        # Calculate values for each category in data
        for category in self.data_lists:
            sample_stdev = stdev(self.data_lists[category])
            sample_mean = mean(self.data_lists[category])

            # Print information about the category:
            output = ('%-25s [%s]: %-2.6f') % (
                'Standard Deviation for ', category, sample_stdev)
            print(output)
            output = ('%-25s [%s]: %-2.6f') % (
                'Mean Deviation for ', category, sample_mean)
            print(output)

    def _bayesian(self, category, height):
        """Create bayesian multiplier.

        Args:
            category: Category of data
            height: Height to process

        Returns:
            value: Multiplier value

        """
        # Initialize key variables
        sample_stdev = stdev(self.data_lists[category])
        sample_mean = mean(self.data_lists[category])

        total = len(self.data_lists[category])
        multiplier = 1 / (math.sqrt(2 * math.pi) * sample_stdev)
        power = math.pow((height - sample_mean) / sample_stdev, 2)
        exponent = math.exp(-0.5 * power)

        # Return
        value = total * multiplier * exponent
        return value

    def _buckets(self, category):
        """Get number of buckets in data.

        Args:
            category: Category of data

        Returns:
            bins: number of buckets

        """
        # Initialize key variables
        interval = 1

        # Get min / max for category
        minimum = int(self.meta[category]['min'])
        maximum = int(self.meta[category]['max'])

        # Calculate bins
        bins = len(range(minimum, maximum, interval))

        # Return
        return (bins, minimum, maximum)

    def _counts(self):
        """Convert file data to probabilty counts.

        Args:
            None

        Returns:
            counts: Dict of counts of height keyed by category and height

        """
        # Initialize key variables
        counts = defaultdict(lambda: defaultdict(dict))

        # Create counts dict for probabilities
        for pair in self.data:
            height = pair[0]
            category = pair[1]

            # Calculate bin
            nbins = self.buckets(category)
            minx = self.minimum(category)
            maxx = self.maximum(category)
            bucket = int(1 + (nbins - 1) * (
                height - minx) / (maxx - minx)) + minx

            # Assign values to bins
            if category in counts:
                if bucket in counts[category]:
                    counts[category][bucket] = counts[
                        category][bucket] + 1
                else:
                    counts[category][bucket] = 1
            else:
                counts[category][bucket] = 1

        # Return
        return counts

    def _lists(self):
        """Convert count data lists.

        Args:
            None

        Returns:
            data: Dict of histogram data keyed by category (gender)

        """
        # Initialize key variables
        data = {}

        # Create counts dict for probabilities
        for pair in self.data:
            height = pair[0]
            category = pair[1]
            if category in data:
                data[category].append(height)
            else:
                data[category] = [height]

        # Return
        return data

    def _probability(self):
        """Calculate probabilities.

        Args:
            category: Category of data
            height: Height to calculate probaility for

        Returns:
            probability: Probabilities dict keyed by category and height

        """
        # Initialize key variables
        counts = self.counts()
        total = defaultdict(lambda: defaultdict(dict))
        probability = defaultdict(lambda: defaultdict(dict))
        icount = defaultdict(lambda: defaultdict(dict))

        # Cycle through data to get gender totals
        for gender in counts.keys():
            for height in counts[gender].keys():
                # Calculate count
                count = counts[gender][height]

                # Get total counts for gender
                if gender in total:
                    total[gender] = total[gender] + count
                else:
                    total[gender] = count

                # Get total counts independent of gender
                if None in total:
                    total[None] = total[None] + count
                else:
                    total[None] = count

                # Get counts independent of gender
                if height in icount:
                    icount[height] = icount[height] + count
                else:
                    icount[height] = count

        # Cycle through data to get gender probabilities
        for gender in counts.keys():
            for height in counts[gender].keys():
                # Do probabilities (category)
                probability[gender][height] = counts[
                    gender][height] / total[gender]

                # Do probabilities (independent)
                probability[None][height] = icount[height] / total[None]

        # Return
        return probability


def _get_gender(male_probability):
    """Determine the gender based on probability.

    Args:
        male_probability: Probability of being male

    Returns:
        gender: Gender

    """
    # Get likely gender
    if male_probability < 0.5:
        gender = 'Female'
    elif male_probability == 0.5:
        gender = 'N/A'
    else:
        gender = 'Male'

    # Return
    return gender
