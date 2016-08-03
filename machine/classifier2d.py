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

# Our library imports
from machine import pca


class Histogram(object):
    """Class for 2 dimensional histogram.

    Args:
        None

    Returns:
        None

    Functions:
        __init__:
        get_cli:
    """

    def __init__(self, pca_object, bins=25):
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
        self.pca_object = pca_object

        # Convert pca_object data to data acceptable by the Histogram2D class
        data = _get_data(pca_object, components=2)

        # Get new PCA object for principal components
        self.pca_new = pca.PCA(data)

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

        # Create empty 2D array
        for cls in values_by_class.keys():
            self.hgram[cls] = np.zeros(
                (self.bin_count, self.bin_count))

        # Get bins data should be placed in
        for cls, tuple_list in sorted(values_by_class.items()):
            for values in tuple_list:
                (row, col) = self._row_col(values)

                # Update histogram
                self.hgram[cls][row][col] += 1

        # Create a list of classes found
        self.classes = sorted(values_by_class.keys())

    def _row_col(self, dimensions):
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

    def accuracy(self):
        """Calulate the accuracy of the training data using histograms.

        Args:
            None

        Returns:
            accuracy: Prediction accuracy

        """
        # Initialize key variables
        correct = {}
        prediction = 0
        cls_count = {}
        accuracy = {}

        # Analyze all the data
        for cls in self.pca_object.classes():
            # Get list of x values to test
            vectors = self.pca_object.xvalues(cls)

            # Process each vector
            for xvalue in vectors:
                # Get prediction
                prediction = self.classifier(xvalue)

                if prediction is not None:
                    # Count the number of correct predictions
                    if prediction == cls:
                        if cls in correct:
                            correct[cls] += 1
                        else:
                            correct[cls] = 1

                    # Increment the count
                    if cls in cls_count:
                        cls_count[cls] += 1
                    else:
                        cls_count[cls] = 1

        # Calculate per class accuracy
        correct[None] = 0
        cls_count[None] = 0
        for cls in cls_count.keys():
            if cls_count[cls] != 0:
                accuracy[cls] = 100 * (correct[cls] / cls_count[cls])

            # Keep a tally for all successes
            correct[None] = correct[None] + correct[cls]
            cls_count[None] = cls_count[None] + cls_count[cls]

        # Calulate overall accuracy
        accuracy[None] = 100 * (correct[None] / cls_count[None])

        # Return
        return accuracy

    def probability(self, xvalue):
        """Get the number of bins to use.

        Args:
            dimensions: Tuple of dimensions

        Returns:
            selection: Class classifier chooses

        """
        # Initialize key variables
        probability = {}

        # Convert X value to principal components
        p1p2 = self.pca_object.pc_of_x(xvalue)

        # Get row / column for histogram for dimensions
        row, col = self._row_col(p1p2)

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

        # Return
        return probability

    def classifier(self, xvalue):
        """Get the number of bins to use.

        Args:
            dimensions: Tuple of dimensions

        Returns:
            selection: Class classifier chooses

        """
        # Initialize key variables
        probability = self.probability(xvalue)

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
            x_col_width = np.ones(num_elements) * self.width(0)
            y_col_depth = np.ones(num_elements) * self.width(1)
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


class Bayesian(object):
    """Class for principal component analysis probabilities.

    Args:
        None

    Returns:
        None

    Functions:
        __init__:
        get_cli:
    """

    def __init__(self, pca_object):
        """Method for intializing the class.

        Args:
            classes: List of classes to process
            pca_object: PCA class object

        Returns:
            None

        """
        # Initialize key variables
        self.components = 2
        self.pca_object = pca_object

        # Convert pca_object data to data acceptable by the Histogram2D class
        self.data = _get_data(pca_object, components=self.components)
        self.class_list = self.pca_object.classes()

        # Get new PCA object for principal components
        self.pca_new = pca.PCA(self.data)

    def classes(self):
        """Get the classes.

        Args:
            cls: Class of data

        Returns:
            value: classes

        """
        # Return
        value = self.class_list
        return value

    def meanvector(self, cls):
        """Get the meanvector.

        Args:
            cls: Class of data

        Returns:
            value: meanvector

        """
        # Return
        value = self.pca_new.meanvector(cls=cls)
        return value

    def covariance(self, cls):
        """Get the covariance.

        Args:
            cls: Class of data

        Returns:
            value: covariance

        """
        # Return
        value = self.pca_new.covariance(cls=cls)
        return value

    def accuracy(self):
        """Calulate the accuracy of the training data using gaussian models.

        Args:
            None

        Returns:
            accuracy: Prediction accuracy

        """
        # Initialize key variables
        correct = {}
        prediction = 0
        cls_count = {}
        accuracy = {}

        # Analyze all the data
        for cls in self.pca_object.classes():
            # Get list of x values to test
            vectors = self.pca_object.xvalues(cls)

            # Process each vector
            for vector in vectors:
                # Get the prediction
                prediction = self.classifier(vector)

                # Only count definitive predictions
                if prediction is not None:
                    # Count the number of correct predictions
                    if prediction == cls:
                        if cls in correct:
                            correct[cls] += 1
                        else:
                            correct[cls] = 1

                    # Increment the count
                    if cls in cls_count:
                        cls_count[cls] += 1
                    else:
                        cls_count[cls] = 1

        # Calculate per class accuracy
        correct[None] = 0
        cls_count[None] = 0
        for cls in cls_count.keys():
            if cls_count[cls] != 0:
                accuracy[cls] = 100 * (correct[cls] / cls_count[cls])

            # Keep a tally for all successes
            correct[None] = correct[None] + correct[cls]
            cls_count[None] = cls_count[None] + cls_count[cls]

        # Calulate overall accuracy
        accuracy[None] = 100 * (correct[None] / cls_count[None])

        # Return
        return accuracy

    def classifier(self, xvalue):
        """Bayesian classifer for any value of X.

        Args:
            xvalue: Specific feature vector of X

        Returns:
            selection: Class classifier chooses

        """
        # Initialize key variables
        probability = {}
        classes = self.classes()

        # Get probability of each class
        probability = self.probability(xvalue)

        # Reassign variables for readability
        prob_c0 = probability[classes[0]]
        prob_c1 = probability[classes[1]]

        # Evaluate probabilities
        if prob_c0 + prob_c1 == 0:
            selection = None
        else:
            if prob_c0 > prob_c1:
                selection = classes[0]
            elif prob_c0 < prob_c1:
                selection = classes[1]
            else:
                selection = None

        # Return
        return selection

    def probability(self, xvalue):
        """Bayesian probability for any value of X.

        Args:
            xvalue: Specific feature vector of X

        Returns:
            selection: Class classifier chooses

        """
        # Initialize key variables
        probability = {}
        bayesian = {}
        classes = self.classes()

        # Calculate the principal components of the individual xvalue
        p1p2 = self.pca_object.pc_of_x(xvalue)

        # Get probability of each class
        for cls in classes:
            # Initialize values for the loop
            sample_count = len(self.pca_object.xvalues(cls))

            # Get values for calculating gaussian parameters
            dimensions = len(p1p2)
            x_mu = p1p2 - self.meanvector(cls)
            covariance = self.covariance(cls)
            inverse_cov = np.linalg.inv(covariance)
            determinant_cov = np.linalg.det(covariance)

            # Work on the exponent part of the bayesian classifer
            power = -0.5 * np.dot(np.dot(x_mu, inverse_cov), x_mu.T)
            exponent = math.pow(math.e, power)

            # Determine the constant value
            pipart = math.pow(2 * math.pi, dimensions / 2)
            constant = pipart * math.sqrt(determinant_cov)

            # Determine final bayesian
            bayesian[cls] = (sample_count * exponent) / constant

        # Calculate bayesian probability
        denominator = bayesian[classes[0]] + bayesian[classes[1]]
        for cls in classes:
            probability[cls] = bayesian[cls] / denominator

        # Return
        return probability


def _get_data(pca_object, components=2):
    """Method for intializing the class.

    Args:
        classes: List of classes to process
        pca_object: PCA class object

    Returns:
        None

    """
    # Initialize key variables
    data = []

    # Convert pca_object data to data acceptable by the Histogram2D class
    (principal_classes,
     principal_components) = pca_object.principal_components(
         components=components)

    for idx, cls in enumerate(principal_classes):
        dimensions = principal_components[idx, :]
        data.append(
            (cls, dimensions.tolist())
        )

    # Return
    return data
