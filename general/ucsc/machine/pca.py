"""Class for principal component analysis."""

# Standard python imports
import sys
import csv
import time
from collections import defaultdict
import math
import operator
from pprint import pprint

# Non-standard python imports
import numpy as np


class PCA(object):
    """Class for principal component analysis.

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
                (class, feature_vector)

        """
        # Initialize key variables
        self.data = data
        self.x_values = {}
        self.pca = defaultdict(lambda: defaultdict(dict))
        class_rows = {}

        # Determine the number of dimensions in vector
        # Create a list of lists
        for cls, vector in data:
            if cls in class_rows:
                class_rows[cls].append(vector)
            else:
                class_rows[cls] = []
                class_rows[cls].append(vector)

        # Create a numpy array for the class
        for cls in class_rows.keys():
            self.x_values[cls] = np.asarray(class_rows[cls])

        # Note the available classes
        self.available_classes = sorted(class_rows.keys())

        # Create a numpy array for the class
        if len(self.x_values.keys()) != 2:
            print('PCA2d class works best with two keys')
            sys.exit(0)

        # Precalculate values
        cls_none = sorted(self.available_classes)
        cls_none.append(None)
        for cls in cls_none:
            # Skip if there are less than two dimensions
            self.pca['xvalues'][cls] = self.xvalues(cls)
            self.pca['meanvector'][cls] = self._meanvector(cls)
            self.pca['zvalues'][cls] = self._zvalues(cls)
            self.pca['covariance'][cls] = self._covariance(cls)

        # Calculate non class specific values
        self.pca['eigenvectors'] = self._eigenvectors()
        self.pca['principal_components'] = self._principal_components()

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

    def classes(self):
        """Return a list of classes in the PCA.

        Args:
            None

        Returns:
            data: List of classes

        """
        # Get xvalues
        data = self.available_classes
        return data

    def xvalues(self, cls=None):
        """Return the input vector array for the input class.

        Args:
            cls: Class of data

        Returns:
            data: X values for the class

        """
        # Get xvalues
        if cls is None:
            data = np.concatenate(
                (self.x_values[self.classes()[0]],
                 self.x_values[self.classes()[1]]))
        else:
            data = self.x_values[cls]
        return data

    def zvalues(self, cls=None):
        """Get the normalized values of ingested data arrays.

        Args:
            cls: Class of data

        Returns:

            self.pca['zvalues'][cls]: zvalues for the class

        """
        # Get zvalues
        return self.pca['zvalues'][cls]

    def meanvector(self, cls=None):
        """Calculate the mean vector of the X array.

        Args:
            cls: Class of data

        Returns:
            self.pca['meanvector'][cls]: meanvector for the class

        """
        # Get meanvector
        return self.pca['meanvector'][cls]

    def covariance(self, cls=None):
        """Get covariance of input data array for a given class.

        Args:
            cls: Class of data

        Returns:
            self.pca['covariance'][cls]: covariance for the class

        """
        # Get covariance
        return self.pca['covariance'][cls]

    def eigenvectors(self, components=None):
        """Get reverse sorted numpy array of eigenvectors for a given class.

        Args:
            cls: Class of data
            components: Number of components to process

        Returns:
            result: Result

        """
        # Get eigenvectors
        eigens = self.pca['eigenvectors']

        # Return first 'components' number of rows
        if components is not None:
            result = eigens[: components:, ]
        else:
            result = eigens
        return result

    def principal_components(self, components=2):
        """Get principal components of input data array for a given class.

        Args:
            cls: Class of data
            components: Number of components to process

        Returns:
            result: principal components

        """
        # Get principal_components
        pcomps = self.pca['principal_components'][1]

        # Return first 'components' number of columns
        if components is not None:
            result = pcomps[:, :components]
        else:
            result = pcomps

        # Get classes
        classes = self.pca['principal_components'][0]

        # Get first "component" number of columns and return
        return (classes, result)

    def meanofz(self, cls=None):
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

    def pc_of_x(self, xvalue, components=2):
        """Create a principal component from a single x value.

        Args:
            xvalue: Specific feature vector of X
            cls: Class to which xvalue belongs
            components: Number of components to process

        Returns:
            p1p2: Principal component of a single value of X

        """
        # Initialize key variables
        meanvector = self.meanvector(None)
        eigenvectors = self.eigenvectors(components=components)

        # Create principal component from next X value
        zvalue = np.subtract(xvalue, meanvector)
        p1p2 = np.dot(zvalue, eigenvectors.T)

        # Return principal components
        return p1p2

    def reconstruct(self, xvalue, components):
        """Reconstruct X based on the principal components generated for it.

        Args:
            xvalue: Specific feature vector of X
            cls: Class to which xvalue belongs
            components: Number of components to process

        Returns:
            result: Reconstructed principal components

        Notes:
            Thanks to: http://glowingpython.blogspot.com/2011/07/
                pca-and-image-compression-with-numpy.html

        """
        # Initialize key variables
        meanvector = self.meanvector(None)
        eigenvectors = self.eigenvectors(components=components)

        # Return
        result = np.dot(self.pc_of_x(
            xvalue, components), eigenvectors) + meanvector
        return result

    def _zvalues(self, cls):
        """Get the normalized values of ingested data arrays.

        Args:
            cls: Class of data

        Returns:
            z_values: Values for the class

        """
        # Get zvalues
        data = self.pca['xvalues'][cls]
        mean_v = self.meanvector(cls)
        z_values = np.subtract(data, mean_v)
        return z_values

    def _meanvector(self, cls):
        """Calculate the mean vector of the X array.

        Args:
            cls: Class of data

        Returns:
            mean_v: Column wise means as nparray

        """
        # Return
        data = self.pca['xvalues'][cls]
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

    def _covariance(self, cls):
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

    def _eigen_values_vectors(self):
        """Get eigen of input data array for a given class.

        Args:
            cls: Class of data

        Returns:
            result: Tuple of (eigenvalues, eigenvectors)
                using real eigenvector values

        """
        # Initialize key variables
        values = np.linalg.eig(self.covariance(None))
        (eigenvalues, eigenvectors) = values
        real_vectors = np.real(eigenvectors)
        result = (eigenvalues, real_vectors)

        # Return
        return result

    def _eigen_tuples(self):
        """Get eigens of input data array for a given class.

        Args:
            sort: Sort by eigenvalue if True

        Returns:
            eig_pairs: Tuples of lists of eigenvalues and eigenvectors.
                Both sorted by eigenvalue.
                ([eigenvalues], [eigenvectors])

        """
        # Initialize key variables
        (eigenvalues, eigenvectors) = self._eigen_values_vectors()

        # Convert numpy arrays of [eigenvalue], [eigenvector] to
        # a list of pairs of tuples
        eig_pairs = [(np.abs(
            eigenvalues[i]), eigenvectors[:, i]) for i in range(
                len(eigenvalues))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=operator.itemgetter(0))
        eig_pairs.reverse()

        # Return
        return eig_pairs

    def _eigenvectors(self):
        """Get reverse sorted numpy array of eigenvectors for a given class.

        Args:
            sort: Sort by eigenvalue if True

        Returns:
            values: nparray of real eigenvectors

        """
        # Initialize key variables
        vector_list = []

        # Proecess data
        eig_pairs = self._eigen_tuples()
        for (_, eigenvector) in eig_pairs:
            vector_list.append(eigenvector)
        values = np.asarray(vector_list)

        # Return
        return values

    def _principal_components(self):
        """Get principal components of input data array for a given class.

        Args:
            None

        Returns:
            result: nparray of real eigenvectors

        """
        # Initialize key variables
        classes = []

        # Start calculations
        z_values = self.zvalues(None)
        eigenvectors = self.eigenvectors()
        result = np.dot(z_values, eigenvectors.T)

        # Get classes represented by each row of X values
        for next_class in self.available_classes:
            next_z = self.zvalues(next_class)
            rows = next_z.shape[0]
            classes.extend([next_class] * rows)

        # Return
        np_classes = np.asarray(classes)
        return (np_classes, result)

    def eigen_vector_check(self):
        """Verify that the eigen vectors are calcualted OK.

        Args:
            None

        Returns:
            matrix: Numpy array of all ones

        """
        # Initialize key variables
        vectors = self.eigenvectors()
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


def image_by_list(body, prefix=''):
    """Create a representative image from ingested data arrays.

    Args:
        body: Body of .pgm image as numpy array of pixels

    Returns:
        None

    """
    # Initialize key variables
    filename = (
        '/home/peter/Downloads/UCSC/%s-test-%s.pgm') % (prefix, time.time())
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
    """Return the color for value.

    Args:
        value: Value to classify
        minmax: Dict of minimum / maximum to use

    Returns:
        shade: Row / Column for histogram

    """
    # Return
    multiplier = 255
    reverse = int(multiplier * (value - minimum) / (maximum - minimum))
    shade = abs(reverse - multiplier)
    return shade
