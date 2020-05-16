#!/usr/bin/env python3
"""Program creates histograms."""

import math
from collections import defaultdict
from statistics import stdev, mean

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


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

    def __init__(self, data):
        """Function for intializing the class.

        Args:
            data: List tuples [(class, value)]

        """
        # Initialize key variables
        self.data = data
        self.meta = defaultdict(lambda: defaultdict(dict))
        self.classes = []

        # Calculate counts
        self.data_lists = self._lists()

        # Get min / max and bins
        for cls, features in sorted(self.data_lists.items()):
            self.meta[cls]['min'] = min(features)
            self.meta[cls]['max'] = max(features)

        # Get list of categories
        self.classes = sorted(self.meta.keys())

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
        for cls in self.meta.keys():
            minimum = self.meta[cls]['min']
            maximum = self.meta[cls]['max']

        # Get range
        for cls in self.meta.keys():
            minimum = min(self.meta[cls]['min'], minimum)
            maximum = max(self.meta[cls]['max'], maximum)

        # Return
        return (int(minimum), int(maximum))

    def buckets(self, cls):
        """Get number of buckets in data.

        Args:
            cls: Category of data

        Returns:
            bins: number of buckets

        """
        # Initialize key variables
        bins = self._buckets(cls)[0]

        # Return
        return bins

    def minimum(self, cls):
        """Get minimum class key value.

        Args:
            cls: Category of data

        Returns:
            data: minimum value

        """
        # Initialize key variables
        data = self._buckets(cls)[1]

        # Return
        return data

    def maximum(self, cls):
        """Get maximum class key value.

        Args:
            cls: Category of data

        Returns:
            data: maximum value

        """
        # Initialize key variables
        data = self._buckets(cls)[2]

        # Return
        return data

    def counts(self):
        """Convert file data to probabilty counts.

        Args:
            None

        Returns:
            data: Dict of counts of feature keyed by class and feature

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
            data: Dict of histogram data keyed by class (gender)

        """
        # Initialize key variables
        data = self.data_lists

        # Return
        return data

    def probability(self, feature, cls=None):
        """Calculate probabilities.

        Args:
            cls: Category of data
            feature: Height to calculate probaility for

        Returns:
            value: Value of probability

        """
        # Initialize key variables
        if cls is None:
            value = self.data_probability[None][feature]
        else:
            value = self.data_probability[cls][feature]

        # Return
        return value

    def table(self, feature_label):
        """Create classifier chart.

        Args:
            feature_label: Feature label

        Returns:
            None

        """
        # Initialize key variables
        (minimum, maximum) = self.bucket_range()
        [cls_0, cls_1] = self.classes

        # Header
        output = ('%10s %15s %15s %15s %15s %6s %6s') % (
            feature_label.capitalize(),
            'Class (H)', 'Prob. (H)',
            'Class (B)', 'Prob. (B)',
            cls_1.capitalize(), cls_0.capitalize())
        print(output)

        # Evaluate features
        for feature in range(minimum, maximum + 1):
            # Get probabilities
            b_probability = self.bayesian_classifier(feature)
            h_probability = self.histogram_classifier(feature)

            # Get genders
            b_class = _get_class(b_probability)
            h_class = _get_class(h_probability)

            # Get counts
            mcount = self.data_counts[cls_1][feature]
            if bool(mcount) is False:
                mcount = 0
            fcount = self.data_counts[cls_0][feature]
            if bool(fcount) is False:
                fcount = 0

            # Account for zero males / females for feature
            if mcount + fcount == 0:
                b_class = 'N/A'
                h_class = 'N/A'

            # Print output
            output = ('%10d %15s %15.6f %15s %15.6f %6d %6d') % (
                feature,
                h_class, h_probability,
                b_class, b_probability,
                mcount, fcount)
            print(output)

    def bayesian_classifier(self, feature):
        """Create histogram classifier chart.

        Args:
            None

        Returns:
            male_probability: Probability of being male

        """
        # Initialize key variables
        [cls_0, cls_1] = self.classes

        # Get class 1 counts (male)
        mcount = self._bayesian(cls_1, feature)

        # Get class 0 counts (female)
        fcount = self._bayesian(cls_0, feature)

        # Get male probability
        if mcount == 0:
            male_probability = 0
        else:
            male_probability = mcount / (mcount + fcount)

        # Return
        return male_probability

    def histogram_classifier(self, feature):
        """Create histogram classifier chart.

        Args:
            None

        Returns:
            None

        """
        # Initialize key variables
        [cls_0, cls_1] = self.classes

        # Get male counts
        if feature not in self.data_counts[cls_1]:
            mcount = 0
        else:
            mcount = self.data_counts[cls_1][feature]

        # Get female counts
        if feature not in self.data_counts[cls_0]:
            fcount = 0
        else:
            fcount = self.data_counts[cls_0][feature]

        # Get male probability
        if mcount == 0:
            male_probability = 0
        else:
            male_probability = mcount / (mcount + fcount)

        # Return
        return male_probability

    def graph(self, feature_label):
        """Graph histogram.

        Args:
            feature_label: Feature label

        Returns:
            None

        """
        # Initialize key variables
        directory = '/home/peter/Downloads'
        data = self.counts()
        categories = []
        features = {}
        counts = {}
        min_feature = 100000000000000000000
        max_feature = 1 - min_feature
        max_count = 1 - 100000000000000000000

        # Create the histogram plot
        fig, axes = plt.subplots(figsize=(7, 7))

        # Random colors for each plot
        prop_iter = iter(plt.rcParams['axes.prop_cycle'])

        # Loop through data
        for cls in data:
            # Create empty list of features
            features[cls] = []
            counts[cls] = []

            # Append cls name
            categories.append(cls.capitalize())

            # Create lists to chart
            for feature, count in sorted(data[cls].items()):
                features[cls].append(feature)
                counts[cls].append(count)

            # Get max / min features
            max_feature = max(max_feature, max(features[cls]))
            min_feature = min(min_feature, min(features[cls]))

            # Get max / min counts
            max_count = max(max_count, max(counts[cls]))

            # Chart line
            plt.plot(
                features[cls], counts[cls],
                color=next(prop_iter)['color'], label=cls.capitalize())

        # Put ticks only on bottom and left
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('left')

        # Set X axis ticks
        major_ticks = np.arange(min_feature, max_feature, 1)
        axes.set_xticks(major_ticks)

        # Set y axis ticks
        major_ticks = np.arange(0, max_count + 100, 50)
        axes.set_yticks(major_ticks)

        # Add legend
        plt.legend()

        # Add Main Title
        fig.suptitle(
            ('%s Histogram') % (feature_label),
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
        axes.set_xlabel(feature_label)

        # Rotate the labels
        for label in axes.xaxis.get_ticklabels():
            label.set_rotation(90)

        # Adjust bottom
        fig.subplots_adjust(left=0.2, bottom=0.2)

        # Create image
        graph_filename = ('%s/homework-%s-%s-rows.png') % (
            directory, feature_label, self.entries)

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

        # Calculate values for each class in data
        for cls in self.data_lists:
            sample_stdev = stdev(self.data_lists[cls])
            sample_mean = mean(self.data_lists[cls])

            # Print information about the class:
            output = ('%-25s [%s]: %-2.6f') % (
                'Standard Deviation for ', cls, sample_stdev)
            print(output)
            output = ('%-25s [%s]: %-2.6f') % (
                'Mean Deviation for ', cls, sample_mean)
            print(output)

    def _bayesian(self, cls, feature):
        """Create bayesian multiplier.

        Args:
            cls: Category of data
            feature: Height to process

        Returns:
            value: Multiplier value

        """
        # Initialize key variables
        sample_stdev = stdev(self.data_lists[cls])
        sample_mean = mean(self.data_lists[cls])

        total = len(self.data_lists[cls])
        multiplier = 1 / (math.sqrt(2 * math.pi) * sample_stdev)
        power = math.pow((feature - sample_mean) / sample_stdev, 2)
        exponent = math.exp(-0.5 * power)

        # Return
        value = total * multiplier * exponent
        return value

    def _buckets(self, cls):
        """Get number of buckets in data.

        Args:
            cls: Category of data

        Returns:
            bins: number of buckets

        """
        # Initialize key variables
        interval = 1

        # Get min / max for class
        minimum = int(self.meta[cls]['min'])
        maximum = int(self.meta[cls]['max'])

        # Calculate bins
        bins = len(range(minimum, maximum, interval))

        # Return
        return (bins, minimum, maximum)

    def _counts(self):
        """Convert file data to probabilty counts.

        Args:
            None

        Returns:
            counts: Dict of counts of feature keyed by class and feature

        """
        # Initialize key variables
        counts = defaultdict(lambda: defaultdict(dict))

        # Create counts dict for probabilities
        for pair in self.data:
            feature = pair[1]
            cls = pair[0]

            # Calculate bin
            nbins = self.buckets(cls)
            minx = self.minimum(cls)
            maxx = self.maximum(cls)
            bucket = int(1 + (nbins - 1) * (
                feature - minx) / (maxx - minx)) + minx

            # Assign values to bins
            if cls in counts:
                if bucket in counts[cls]:
                    counts[cls][bucket] = counts[
                        cls][bucket] + 1
                else:
                    counts[cls][bucket] = 1
            else:
                counts[cls][bucket] = 1

        # Return
        return counts

    def _lists(self):
        """Convert count data lists.

        Args:
            None

        Returns:
            data: Dict of histogram data keyed by class (gender)

        """
        # Initialize key variables
        data = {}

        # Create counts dict for probabilities
        for pair in self.data:
            feature = pair[0]
            cls = pair[1]
            if cls in data:
                data[cls].append(feature)
            else:
                data[cls] = [feature]

        # Return
        return data

    def _probability(self):
        """Calculate probabilities.

        Args:
            feature: Height to calculate probaility for

        Returns:
            probability: Probabilities dict keyed by class and feature

        """
        # Initialize key variables
        counts = self.counts()
        total = defaultdict(lambda: defaultdict(dict))
        probability = defaultdict(lambda: defaultdict(dict))
        icount = defaultdict(lambda: defaultdict(dict))

        # Cycle through data to get class totals
        for cls in counts.keys():
            for feature in counts[cls].keys():
                # Calculate count
                count = counts[cls][feature]

                # Get total counts for gender
                if cls in total:
                    total[cls] = total[cls] + count
                else:
                    total[cls] = count

                # Get total counts independent of gender
                if None in total:
                    total[None] = total[None] + count
                else:
                    total[None] = count

                # Get counts independent of gender
                if feature in icount:
                    icount[feature] = icount[feature] + count
                else:
                    icount[feature] = count

        # Cycle through data to get gender probabilities
        for cls in counts.keys():
            for feature in counts[cls].keys():
                # Do probabilities (class)
                probability[cls][feature] = counts[
                    cls][feature] / total[cls]

                # Do probabilities (independent)
                probability[None][feature] = icount[feature] / total[None]

        # Return
        return probability


def _get_class(male_probability):
    """Determine the gender based on probability.

    Args:
        male_probability: Probability of being male

    Returns:
        gender: Gender

    """
    # Get likely gender
    if male_probability < 0.5:
        cls = 'Female'
    elif male_probability == 0.5:
        cls = 'N/A'
    else:
        cls = 'Male'

    # Return
    return cls
