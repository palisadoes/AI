#!/usr/bin/env python3
"""Program creates histograms."""

from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


class Chart(object):
    """Class gathers all CLI information.

    Args:
        None

    Returns:
        None

    Functions:
        __init__:
        get_cli:
    """

    def __init__(self, classes, data):
        """Function for intializing the class.

        Args:
            data: Tuple (class, dimension, dimension)

        """
        # Initialize key variables
        self.classes = classes
        self.data = data

    def graph(self):
        """Graph histogram.

        Args:
            feature_label: Feature label

        Returns:
            None

        """
        # Initialize key variables
        directory = '/home/peter/Downloads'
        display_data = defaultdict(lambda: defaultdict(dict))

        # Prepopulate lists of data to display
        for cls in self.classes:
            display_data[cls][0] = []
            display_data[cls][1] = []

        # Create the histogram plot
        fig, axes = plt.subplots(figsize=(7, 7))

        # Random colors for each plot
        prop_iter = iter(plt.rcParams['axes.prop_cycle'])

        # Loop through data to create chartable lists by class
        class_array = self.data[0]
        p1_values = self.data[1]
        p2_values = self.data[2]
        for (col, ), cls in np.ndenumerate(class_array):
            display_data[cls][0].append(p1_values[col])
            display_data[cls][1].append(p2_values[col])

        # Start plotting data
        for cls in self.classes:

            # Get color of plot
            color = next(prop_iter)['color']

            # Create plot
            plt.scatter(
                display_data[cls][0],
                display_data[cls][1],
                c=color, alpha=0.5)

        # Create image
        graph_filename = ('%s/homework-scatter.png') % (directory)

        # Save chart
        fig.savefig(graph_filename)

        # Close the plot
        plt.close(fig)
