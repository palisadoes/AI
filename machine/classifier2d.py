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

from machine import pca
from machine import histogram2d


class Classifier2D(object):
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
            data: List of principal component tuples of format
                (class, feature_vector)

        """
        # Initialize key variables
        self.labels = labels
        self.hgram = {}
        self.minmax = defaultdict(lambda: defaultdict(dict))
        values_by_class = defaultdict(lambda: defaultdict(dict))
        self.x_y = defaultdict(lambda: defaultdict(dict))

        # Create PCA object for principal components
        self.pca_object = pca.PCA(data)

        # Create Histogram2d object for principal components
        self.h_object = histogram2d.Histogram2d(data)

class Probability2D(object):
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
        self.data = []

        # Get classes
        self.classes = self.pca_object.classes()

        # Retrieve principal components and their associated classes
        (principal_classes,
         principal_components) = pca_object.principal_components(
             components=components)

        # Save principal components for later use
        pc_1 = principal_components[:, 0]
        pc_2 = principal_components[:, 1]

        # Loop through data to create chartable lists by class
        new_data = []
        for (col, ), cls in np.ndenumerate(principal_classes):
            new_stack = np.array(np.hstack((pc_1[col], pc_2[col])))
            new_data.append(
                (cls, new_stack)
            )
        h_object = histogram2d.Histogram2d(new_data)



        # Convert pca_object data to data acceptable by the Histogram2D class
        for cls in self.classes:
            principal_components = self.pca_object.principal_components(
                cls, components=self.components)[1]
            for dimension in principal_components:
                self.data.append(
                    (cls, dimension[0], dimension[1])
                )

        # Get histogram
        self.hist_object = histogram2d.Histogram2D(self.data, self.classes)

    def histogram(self):
        """Get the histogram.

        Args:
            None

        Returns:
            value: 2D histogram

        """
        # Return
        value = self.hist_object.histogram()
        return value

    def histogram_accuracy(self):
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
        for item in self.data:
            cls = item[0]
            dim0 = item[1]
            dim1 = item[2]

            # Get the prediction
            values = (dim0, dim1)
            prediction = self.histogram_prediction(values, cls)

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

        # Return
        for cls in sorted(cls_count.keys()):
            accuracy[cls] = 100 * (correct[cls] / cls_count[cls])
        return accuracy

    def histogram_prediction(self, values, cls):
        """Calulate the accuracy of the training data using histograms.

        Args:
            values: Tuple from which to create the principal components
            cls: Class of data to which data belongs

        Returns:
            prediction: Class of prediction

        """
        # Get the principal components for data
        pc_values = self.pca_object.pc_of_x(values, cls, self.components)
        # print('ppppp', values, pc_values, type(pc_values))

        # Get row / column for histogram for principal component
        prediction = self.hist_object.classifier(pc_values)

        # Return
        return prediction

    def gaussian_accuracy(self):
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
        for item in self.data:
            cls = item[0]
            dim0 = item[1]
            dim1 = item[2]

            # Get the prediction
            values = (dim0, dim1)
            prediction = self.gaussian_prediction(values, cls)

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

        # Return
        for cls in sorted(cls_count.keys()):
            accuracy[cls] = 100 * (correct[cls] / cls_count[cls])
        return accuracy

    def gaussian_prediction(self, values, cls):
        """Predict class using gaussians models.

        Args:
            values: Tuple from which to create the principal components
            cls: Class of data to which data belongs

        Returns:
            prediction: Class of prediction

        """
        # Initialize key variables
        pc_values = []

        # Get the principal components for data
        pc_values.append(
            self.pca_object.pc_of_x(values, cls, self.components))

        # Get row / column for histogram for principal component
        prediction = self.pca_object.classifier2d(np.asarray(pc_values))

        # Return
        return prediction
