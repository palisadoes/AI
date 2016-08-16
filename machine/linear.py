"""Class for principal component analysis."""

# Standard python imports

# Non-standard python imports
import numpy as np


class Linear(object):
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
            data: Kessler training data
                (list of lists, first column is always 1)

        """
        # Initialize key variables
        self.data = data

    def classifier(self, classes):
        """Create binary linear classifier.

        Args:
            classes: List of class definitions for training data

        Returns:
            result: Classifier

        """
        # Initialize key variables
        pseudo = np.linalg.pinv(self.data)
        result = np.dot(pseudo, classes)
        return result

    def prediction(self, vector, classes):
        """Predict the class of the vector.

        Args:
            vector: Vector

        Returns:
            result: Class of prediction

        """
        # Initialize key variables
        classification = np.dot(vector, self.classifier(classes))
        (columns,) = classification.shape

        # Make the prediction
        if columns == 1:
            # Binary classifier
            if classification[0] > 0:
                result = 1
            else:
                result = -1
        else:
            # Non-binary classifier
            values = classification.tolist()
            maximum = max(values)
            result = values.index(maximum)

        return result
