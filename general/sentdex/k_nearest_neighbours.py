#!/usr/bin/env python3
"""Script to demonstrate  K nearest neighbours machine learning."""

import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd


def main():
    """Main Function.

    Display data prediction from K nearest neighbours model

    The data file breast-cancer-wisconsin.data has the following format:

       #  Attribute                     Domain
       -- -----------------------------------------
       1. Sample code number            id number
       2. Clump Thickness               1 - 10
       3. Uniformity of Cell Size       1 - 10
       4. Uniformity of Cell Shape      1 - 10
       5. Marginal Adhesion             1 - 10
       6. Single Epithelial Cell Size   1 - 10
       7. Bare Nuclei                   1 - 10
       8. Bland Chromatin               1 - 10
       9. Normal Nucleoli               1 - 10
      10. Mitoses                       1 - 10
      11. Class:                        (2 for benign, 4 for malignant)

    The data was obtained from:

        https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

    We wiil add the following header to the file:

    id,clump_thickness,uniform_cell_size,uniform_cell_shape,marginal_adhesion,
    single_epi_cell_size,bare_nuclei,bland_chromation,normal_nucleoli,mitoses,
    class

    """
    # Initialize key variables
    labels = [
        'id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
        'marginal_adhesion', 'single_epi_cell_size', 'bare_nuclei',
        'bland_chromation', 'normal_nucleoli', 'mitoses', 'class']
    filename = 'data/breast-cancer-wisconsin.data'
    test_size_percent = 20

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
    clf = neighbors.KNeighborsClassifier()
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
