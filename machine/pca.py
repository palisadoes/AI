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
        self.class_list = self.pca_object.classes()

        # Convert pca_object data to data acceptable by the Histogram2D class
        (principal_classes,
         principal_components) = self.pca_object.principal_components(
             components=self.components)

        for idx, cls in enumerate(principal_classes):
            dimensions = principal_components[idx, :]
            self.data.append(
                (cls, dimensions.tolist())
            )

        # Get new PCA object for principal components
        self.pca_new = PCA(self.data)

        # Get histogram
        self.hist_object = histogram2d.Histogram2D(self.data)

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

    def classifier_histogram(self, p1p2):
        """Get the classifier_histogram.

        Args:
            p1p2: Principal components

        Returns:
            value: classifier_histogram

        """
        # Return
        value = self.hist_object.classifier_histogram(p1p2)
        return value

    def probability_histogram(self, p1p2):
        """Get the probability_histogram.

        Args:
            p1p2: Principal components

        Returns:
            value: probability_histogram

        """
        # Return
        value = self.hist_object.probability_histogram(p1p2)
        return value

    def accuracy_histogram(self):
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
            for vector in vectors:
                # Calculate the principal components of the individual xvalue
                p1p2 = self.pca_object.pc_of_x(vector)

                # Get prediction
                prediction = self.classifier_histogram(p1p2)

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

    def accuracy_bayesian(self):
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
                prediction = self.classifier_bayesian(vector)

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
        print('help', correct[None], cls_count[None])
        accuracy[None] = 100 * (correct[None] / cls_count[None])

        # Return
        return accuracy

    def classifier_bayesian(self, xvalue):
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
        probability = self.probability_bayesian(xvalue)

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

    def probability_bayesian(self, xvalue):
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

        # Get probability of each class
        for cls in classes:
            # Initialize values for the loop
            sample_count = len(self.pca_object.xvalues(cls))

            # Calculate the principal components of the individual xvalue
            p1p2 = self.pca_object.pc_of_x(xvalue)

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
