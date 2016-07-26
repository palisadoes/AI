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

    def __init__(self, data):
        """Function for intializing the class.

        Args:
            data: List tuples [(class, dimension, dimension)]

        """
        # Initialize key variables
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

        # Create the histogram plot
        fig, axes = plt.subplots(figsize=(7, 7))

        # Random colors for each plot
        prop_iter = iter(plt.rcParams['axes.prop_cycle'])

        # Loop through data
        for item in sorted(self.data):
            cls = item[0]
            p1_values = item[1]
            p2_values = item[2]

            # Get color of plot
            color = next(prop_iter)['color']

            # Create plot
            plt.scatter(p1_values, p2_values, c=color, alpha=0.5)

        # Create image
        graph_filename = ('%s/homework-scatter.png') % (directory)

        # Save chart
        fig.savefig(graph_filename)

        # Close the plot
        plt.close(fig)
