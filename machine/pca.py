#!/usr/bin/env python3
"""Program creates histograms."""

# Standard python imports
import sys
import csv
import time
from collections import defaultdict
import math
from pprint import pprint

# Non-standard python imports
import numpy as np


class PCA2d(object):
    """Class for 2 dimensional histogram.

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
            data: List of tuples of format
                (class, dimension1, dimension2 ...)

        """
        # Initialize key variables
        self.data = data
        class_rows = {}
        self.x_values = {}
        minmax = defaultdict(lambda: defaultdict(dict))
        values_by_class = defaultdict(lambda: defaultdict(dict))
        self.x_y = defaultdict(lambda: defaultdict(dict))

        # Determine the number of dimensions in vector
        for cls, vector in data:
            if cls in class_rows:
                class_rows[cls].append(vector)
            else:
                class_rows[cls] = [vector]

        # Create a numpy array for the class
        for cls in class_rows.keys():
            self.x_values[cls] = np.asarray(class_rows[cls])

        # Create a numpy array for the class
        if len(self.x_values.keys()) != 2:
            print('PCA2d class works best with two keys')
            sys.exit(0)

        for cls in self.x_values.keys():
            print(self.x_values[cls].shape)

    def image(self, cls, pointer):
        """Create a representative image from ingested data arrays.

        Args:
            cls: Class of data
            pointer: Pointer to bytes representing image

        Returns:
            None

        """
        # Initialize key variables
        body = self.x_values[cls][pointer]

        # Create final image
        image_by_list(body)

    def meanvector(self, cls):
        """Get the column wise means of ingested data arrays.

        Args:
            cls: Class of data

        Returns:
            mean_v: Column wise means as nparray

        """
        # Return
        data = self.x_values[cls]
        mean_v = data.mean(axis=0)
        return mean_v

    def zvalues(self, cls):
        """Get the normalized values of ingested data arrays.

        Args:
            cls: Class of data

        Returns:
            z_values: Normalized values

        """
        # Get zvalues
        data = self.x_values[cls]
        mean_v = self.meanvector(cls)
        z_values = np.subtract(data, mean_v)
        return z_values

    def meanofz(self, cls):
        """Get mean vector of Z. This is a test, result must be all zeros.

        Args:
            cls: Class of data

        Returns:
            z_values: Normalized values

        """
        # Get values
        data = self.zvalues(cls)
        mean_v = data.mean(axis=0)
        return mean_v

    def covariance(self, cls):
        """Get covariance of input data array for a given class.

        Args:
            cls: Class of data

        Returns:
            matrix: Covariance matrix

        """
        # Initialize key variables
        z_values = self.zvalues(cls)
        (rows, columns) = z_values.shape
        matrix = np.zeros(shape=(columns, columns))

        # Iterate
        for (row, column), _ in np.ndenumerate(z_values):
            # Sum multiplying
            summation = 0
            for ptr_row in range(0, rows):
                summation = summation + (
                    z_values[ptr_row, row] * z_values[ptr_row, column])
            matrix[row, column] = summation / (columns - 1)

        # Return
        return matrix

    def covariance_quick(self, cls):
        """Get covariance of input data array for a given class.

        Args:
            cls: Class of data

        Returns:
            matrix: Covariance matrix

        """
        # Initialize key variables
        stack = np.vstack(self.zvalues(cls))
        matrix = np.cov(stack)

        # Return
        return matrix


def image_by_list(body):
    """Create a representative image from ingested data arrays.

    Args:
        body: Body of .pgm image as numpy array of pixels

    Returns:
        None

    """
    # Initialize key variables
    filename = ('/home/peter/Downloads/test-%s.pgm') % (int(time.time()))
    final_image = []
    body_as_list = body.astype(int).flatten().tolist()

    # Create header
    rows = int(math.sqrt(len(body_as_list)))
    columns = rows
    header = ['P2', rows, columns, 255]

    # Create final image
    final_image.extend(header)
    final_image.extend(body_as_list)

    # Save file
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\n')
        spamwriter.writerow(final_image)
