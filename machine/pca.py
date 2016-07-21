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

    def image_by_vector(self, vector):
        """Create a representative image from a vector.

        Args:
            vector: Vector

        Returns:
            None

        """
        # Create final image
        image_by_list(vector)

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

    def xvalues(self, cls):
        """Return the input vector array for the input class.

        Args:
            cls: Class of data

        Returns:
            data: Normalized values

        """
        # Get xvalues
        data = self.x_values[cls]
        return data

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

    def covariance_manual(self, cls):
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

        # Iterate over the matrix
        for (row, column), _ in np.ndenumerate(matrix):
            # Sum multiplying
            summation = 0
            for ptr_row in range(0, rows):
                summation = summation + (
                    z_values[ptr_row, row] * z_values[ptr_row, column])
            matrix[row, column] = summation / (columns - 1)

        # Return
        return matrix

    def covariance(self, cls):
        """Get covariance of input data array for a given class.

        Args:
            cls: Class of data

        Returns:
            matrix: Covariance matrix

        """
        # Initialize key variables
        zmatrix = self.zvalues(cls).T
        matrix = np.cov(zmatrix)
        return matrix

    def eigen_values_vectors(self, cls, imaginary=False):
        """Get eigen of input data array for a given class.

        Args:
            cls: Class of data

        Returns:
            result: tuple of (eigenvalues, eigenvectors)

        """
        # Initialize key variables
        values = np.linalg.eig(self.covariance(cls))
        if imaginary is False:
            (eigenvalues, eigenvectors) = values
            real_vectors = np.real(eigenvectors)
            result = (eigenvalues, real_vectors)
        else:
            result = values

        # Return
        return result

    def eigen_vectors(self, cls):
        """Get eigen of input data array for a given class.

        Args:
            cls: Class of data

        Returns:
            values: nparray of real eigenvectors

        """
        # Initialize key variables
        values = np.real(self.eigen_values_vectors(cls)[1])

        # Return
        return values

    def principal_components(self, cls):
        """Get eigen of input data array for a given class.

        Args:
            cls: Class of data

        Returns:
            result: nparray of real eigenvectors

        """
        # Initialize key variables
        z_values = self.zvalues(cls)
        (eigenvalues, eigenvectors) = self.eigen_values_vectors(cls)
        result = numpy.dot(z_values, eigenvectors.T)
        return (result, eigenvalues)

    def eigen_vector_check(self, cls):
        """Verify that the eigen vectors are calcualted OK.

        Args:
            cls: Class of data

        Returns:
            matrix: Numpy array of all ones

        """
        # Initialize key variables
        vectors = self.eigen_vectors(cls)
        (_, columns) = vectors.shape
        matrix = np.zeros(shape=(1, columns))

        # Iterate over the matrix
        for column in range(0, columns):
            column_array = vectors[:, column]

            column_sum = 0
            for item in column_array:
                column_sum = column_sum + (item * item)

            # Get square root of column_sum
            matrix[0, column] = math.sqrt(column_sum)

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
    filename = ('/home/peter/Downloads/UCSC/test-%s.pgm') % (time.time())
    final_image = []
    maxshade = int(255)
    body_as_list = body.astype(float).flatten().tolist()
    new_list = [0] * len(body_as_list)

    # Create header
    rows = int(math.sqrt(len(body_as_list)))
    columns = rows
    header = ['P2', rows, columns, maxshade]

    # Normalize values from 1 - 255
    minval = min(body_as_list)
    maxval = max(body_as_list)
    for pointer, value in enumerate(body_as_list):
        shade = _shade(value, minval, maxval)
        new_list[pointer] = shade

    # Create final image
    final_image.extend(header)
    final_image.extend(new_list)

    # Save file
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\n')
        spamwriter.writerow(final_image)


def _shade(value, minimum, maximum):
    """Get the row or column for 2D histogram.

    Args:
        value: Value to classify
        minmax: Dict of minimum / maximum to use

    Returns:
        hbin: Row / Column for histogram

    """
    # Return
    multiplier = 254
    hbin = int(multiplier * (value - minimum) / (maximum - minimum))
    # print(hbin, value, minimum, maximum)
    return hbin
