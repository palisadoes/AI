#!/usr/bin/env python3
"""Script to demonstrate  SVM machine learning."""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class SVM(object):
    """Support vector machine class."""

    def __init__(self, visualization=True):
        """Method to instantiate the class.

        Args:
            visualization: Create image if True

        Returns:
            None

        """
        # Initialize key variables
        self._visualization = visualization
        self._colors = {1: 'r', -1: 'b'}
        self._data = None

        # Set up chart figure
        if self._visualization:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(1, 1, 1)

    def fit(self, data):
        """Train the data.

        Args:
            data: Data to train with. Vectors keyed by class.

        Returns:
            None

        """
        # Initialize key variables
        self._data = data
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        """
        This dictionary will be keyed by the magnitude of w and will contain a
        list of w and its corresponding b.

        { ||w||: [w, b] }

        """
        opt_dict = {}

        # Find values to work with for our ranges.
        # yi is the class class
        all_data = []
        for yi in self._data:
            for featureset in self._data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        # Get the min and max of ALL features in the data set
        self._max_feature_value = max(all_data)
        self._min_feature_value = min(all_data)

        # No need to keep this memory.
        all_data = None

        # Get the step sizes for the vector (w) minimum calculation
        step_sizes = [
            self._max_feature_value * 0.1,
            self._max_feature_value * 0.01,
            # starts getting very high cost after this.
            self._max_feature_value * 0.001
            ]

        # Get the step sizes for the bias (b) minimum calculation
        # Extremely expensive
        b_range_multiple = 5

        # We dont need to take as small of steps with b as we do w
        b_multiple = 5

        # Set the starting poing for the first convex step
        latest_optimum = self._max_feature_value * 10
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # We can do this because convex
            optimized = False

            # Calculate the range of b values to use in the Optimization
            b_range = np.arange(
                -1 * (self._max_feature_value * b_range_multiple),
                self._max_feature_value * b_range_multiple,
                step * b_multiple)

            while not optimized:
                for b in b_range:

                    for transformation in transforms:
                        # Transform w
                        w_t = w * transformation

                        found_option = True
                        # Weakest link in SVM (Iterating over all data)
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        #
                        # #### add a break here later..
                        for next_class in self._data:
                            for xi in self._data[next_class]:
                                yi = next_class
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    # Subtract step from each value in the w vector
                    w = [_ - step for _ in w]

            #||w|| : [w,b]
            # Get the dict with the smallest w magnitude (||w||)
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + (step * 2)

    def predict(self, features):
        """Train the data using Sequential Minimal Optimization (SMO).

        Args:
            features: List of features to use for a prediction

        Returns:
            None

        """
        # sign( x.w+b )
        classification = np.sign(
            np.dot(np.array(features), self.w) + self.b)

        return classification


def main():
    """Main Function.

    Display data prediction from SVM model

    """
    # Initialize key variables
    data_dict = {
        -1: np.array([[1, 7], [2, 8], [3, 8], ]),
        1: np.array([[5, 1], [6, -1], [7, 3], ])
    }

    # Create a sample vector for prediction
    example_measures_1 = [4, 2, 1, 1, 1, 2, 3, 2, 1]
    example_measures_2 = [4, 2, 1, 2, 2, 2, 3, 2, 1]

    # Get data and replace missing values which are denoted by '?' with an
    # outlier value
    dataframe = pd.read_csv(filename, names=labels)
    dataframe.replace('?', -99999, inplace=True)

    # Drop the id row, which is just a row number equivalent
    dataframe.drop(['id'], 1, inplace=True)

    # Create a vector and class array and the training data
    x_ary = np.array(dataframe.drop(['class'], 1))
    y_ary = np.array(dataframe['class'])
    (X_train, X_test, y_train, y_test) = model_selection.train_test_split(
        x_ary, y_ary, test_size=test_size_percent/100)

    # Create the classifier
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    # Print results
    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    # Print the prediction
    # After reshaping the data to a list of lists
    data = [example_measures_1]
    vector_array = np.array(data)
    rows = len(data)
    prediction = clf.predict(vector_array.reshape(rows, -1))
    print(prediction)

    data = [example_measures_1, example_measures_2]
    vector_array = np.array(data)
    rows = len(data)
    prediction = clf.predict(vector_array.reshape(rows, -1))
    print(prediction)


if __name__ == "__main__":
    main()
