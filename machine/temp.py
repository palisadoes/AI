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
