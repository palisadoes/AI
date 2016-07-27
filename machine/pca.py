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

from machine import histogram2d


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
        class_rows = {}
        self.x_values = {}
        self.pca = defaultdict(lambda: defaultdict(dict))

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

        # Precalculate values
        for cls in class_rows.keys():
            self.pca['xvalues'][cls] = self.xvalues(cls)
            self.pca['meanvector'][cls] = self._meanvector(cls)
            self.pca['zvalues'][cls] = self._zvalues(cls)
            self.pca['covariance'][cls] = self._covariance(cls)
            self.pca['eigenvectors'][cls] = self._eigenvectors(
                cls, sort=False)
            self.pca['eigenvectors_sorted'][cls] = self._eigenvectors(
                cls, sort=True)
            self.pca['principal_components'][
                cls] = self._principal_components(cls)

        # Note the available classes
        self.available_classes = sorted(class_rows.keys())

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

    def xvalues(self, cls):
        """Return the input vector array for the input class.

        Args:
            cls: Class of data

        Returns:
            data: X values for the class

        """
        # Get xvalues
        data = self.x_values[cls]
        return data

    def zvalues(self, cls):
        """Get the normalized values of ingested data arrays.

        Args:
            cls: Class of data

        Returns:

            self.pca['zvalues'][cls]: zvalues for the class

        """
        # Get zvalues
        return self.pca['zvalues'][cls]

    def meanvector(self, cls):
        """Calculate the mean vector of the X array.

        Args:
            cls: Class of data

        Returns:
            self.pca['meanvector'][cls]: meanvector for the class

        """
        # Get meanvector
        return self.pca['meanvector'][cls]

    def covariance(self, cls):
        """Get covariance of input data array for a given class.

        Args:
            cls: Class of data

        Returns:
            self.pca['covariance'][cls]: covariance for the class

        """
        # Get covariance
        return self.pca['covariance'][cls]

    def eigenvectors(self, cls, sort=False, components=None):
        """Get reverse sorted numpy array of eigenvectors for a given class.

        Args:
            cls: Class of data
            components: Number of components to process

        Returns:
            result: Result

        """
        # Get eigenvectors
        if sort is False:
            eigens = self.pca['eigenvectors'][cls]
        else:
            eigens = self.pca['eigenvectors_sorted'][cls]

        # Return first 'components' number of rows
        if components is not None:
            result = eigens[: components:, ]
        else:
            result = eigens
        return result

    def principal_components(self, cls, components=None):
        """Get principal components of input data array for a given class.

        Args:
            cls: Class of data
            components: Number of components to process

        Returns:
            result: principal components

        """
        # Get principal_components
        pcomps = self.pca['principal_components'][cls]

        # Return first 'components' number of columns
        if components is not None:
            result = pcomps[:, :components]
        else:
            result = pcomps
        return result

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

    def pc_of_x(self, xvalue, cls, components):
        """Create a principal component from a single x value.

        Args:
            xvalue: Specific feature vector of X
            cls: Class to which xvalue belongs
            components: Number of components to process

        Returns:
            p1p2: Principal component of a single value of X

        """
        # Initialize key variables
        meanvector = self.meanvector(cls)
        eigenvectors = self.eigenvectors(
            cls, sort=True, components=components)

        # Create principal component from next X value
        zvalue = np.subtract(xvalue, meanvector)
        p1p2 = np.dot(zvalue, eigenvectors.T)

        # Return principal components
        return p1p2

    def reconstruct(self, xvalue, cls, components):
        """Reconstruct X based on the principal components generated for it.

        Args:
            xvalue: Specific feature vector of X
            cls: Class to which xvalue belongs
            components: Number of components to process

        Returns:
            result: Reconstructed principal components

        """
        # Initialize key variables
        meanvector = self.meanvector(cls)
        eigenvectors = self.eigenvectors(
            cls, sort=True, components=components)

        # Return
        result = np.dot(self.pc_of_x(
            xvalue, cls, components), eigenvectors) + meanvector
        return result

    def classifier2d(self, xvalue):
        """Bayesian classifer for any value of X.

        Args:
            xvalue: Specific feature vector of X

        Returns:
            selection: Class classifier chooses

        """
        # Initialize key variables
        probability = {}
        bayesian = {}
        classes = self.classes()

        # Get probability of each class
        for cls in classes:
            # Initialize values for the loop
            sample_count = len(self.xvalues(cls))
            x_mu = xvalue - self.meanvector(cls)
            covariance = self.covariance(cls)
            inverse_cov = np.linalg.inv(covariance)
            determinant_cov = np.linalg.det(covariance)
            dimensions = len(xvalue)

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

        # Get selection
        if probability[classes[0]] > probability[classes[1]]:
            selection = classes[0]
        elif probability[classes[0]] < probability[classes[1]]:
            selection = classes[1]
        else:
            selection = None

        # Return
        return selection

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

    def _eigen_values_vectors(self, cls):
        """Get eigen of input data array for a given class.

        Args:
            cls: Class of data

        Returns:
            result: Tuple of (eigenvalues, eigenvectors)
                using real eigenvector values

        """
        # Initialize key variables
        values = np.linalg.eig(self.covariance(cls))
        (eigenvalues, eigenvectors) = values
        real_vectors = np.real(eigenvectors)
        result = (eigenvalues, real_vectors)

        # Return
        return result

    def _eigen_tuples(self, cls, sort=True):
        """Get eigens of input data array for a given class.

        Args:
            cls: Class of data
            sort: Sort by eigenvalue if True

        Returns:
            eig_pairs: Tuples of lists of eigenvalues and eigenvectors.
                Both sorted by eigenvalue.
                ([eigenvalues], [eigenvectors])

        """
        # Initialize key variables
        (eigenvalues, eigenvectors) = self._eigen_values_vectors(cls)

        # Convert numpy arrays of [eigenvalue], [eigenvector] to
        # a list of pairs of tuples
        eig_pairs = [(np.abs(
            eigenvalues[i]), eigenvectors[:, i]) for i in range(
                len(eigenvalues))]

        if sort is True:
            # Sort the (eigenvalue, eigenvector) tuples from high to low
            eig_pairs.sort(key=operator.itemgetter(0))
            eig_pairs.reverse()

        # Return
        return eig_pairs

    def _eigenvectors(self, cls, sort=True):
        """Get reverse sorted numpy array of eigenvectors for a given class.

        Args:
            cls: Class of data
            sort: Sort by eigenvalue if True

        Returns:
            values: nparray of real eigenvectors

        """
        # Initialize key variables
        vector_list = []

        # Proecess data
        eig_pairs = self._eigen_tuples(cls, sort=sort)
        for (_, eigenvector) in eig_pairs:
            vector_list.append(eigenvector)
        values = np.asarray(vector_list)

        # Return
        return values

    def _principal_components(self, cls):
        """Get principal components of input data array for a given class.

        Args:
            cls: Class of data

        Returns:
            result: nparray of real eigenvectors

        """
        # Initialize key variables
        z_values = self.zvalues(cls)
        eigenvectors = self.eigenvectors(cls, sort=True)
        result = np.dot(z_values, eigenvectors.T)
        return result

    def eigen_vector_check(self, cls):
        """Verify that the eigen vectors are calcualted OK.

        Args:
            cls: Class of data

        Returns:
            matrix: Numpy array of all ones

        """
        # Initialize key variables
        vectors = self.eigenvectors(cls)
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

        # Convert pca_object data to data acceptable by the Histogram2D class
        for cls in self.classes:
            principal_components = self.pca_object.principal_components(
                cls, components=self.components)
            for dimension in principal_components:
                self.data.append(
                    (cls, dimension[0], dimension[1])
                )

        # Get histogram
        self.hist_object = histogram2d.Histogram2D(self.data, self.classes)

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
        # Initialize key variables
        pc_values = []

        # Calculate each principal component
        for value in values:
            # Get the principal components for data
            pc_values.append(
                self.pca_object.pc_of_x(value, cls, self.components))

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

        # Calculate each principal component
        for value in values:
            # Get the principal components for data
            pc_values.append(
                self.pca_object.pc_of_x(value, cls, self.components))

        # Get row / column for histogram for principal component
        prediction = self.pca_object.classifier2d(pc_values)

        # Return
        return prediction


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
