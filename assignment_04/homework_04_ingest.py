#!/usr/bin/env python3
"""Program Ingests data for a 2D histogram."""

# Standard python imports
import sys
import argparse
from pprint import pprint
import numpy as np
from sklearn.metrics import confusion_matrix

import xlrd

# Import AI library
from machine.linear import Linear


class Ingest(object):
    """Class for reading in data file.

    Args:
        None

    Returns:
        None

    """

    def __init__(self, filename):
        """Function for intializing the class.

        Args:
            filename: Excel filename to read

        """
        # Initialize key variables
        self.filename = filename

        self.data = self._data()

    def training_data(self):
        """Method to obtain training data.

        Args:
            None

        Returns:
            None

        """
        (data, _, _, _) = self.data
        return np.asarray(data[0])

    def data_to_classify(self):
        """Method to obtain training data.

        Args:
            None

        Returns:
            None

        """
        (data, _, _, _) = self.data
        return np.asarray(data[2])

    def klasses_binary(self):
        """Method to obtain kesslerized classes.

        Args:
            None

        Returns:
            None

        """
        (_, data, _, _) = self.data
        return np.asarray(data)

    def klasses_non_binary(self):
        """Method to obtain training data.

        Args:
            None

        Returns:
            None

        """
        (_, _, data, _) = self.data
        return np.asarray(data)

    def classes_non_binary(self):
        """Method to obtain training data.

        Args:
            None

        Returns:
            None

        """
        (_, _, _, data) = self.data
        return np.asarray(data)

    def _data(self):
        """Method to read data from spreadsheet.

        Args:
            sheet: Number of the sheet to read

        Returns: None

        """
        # Initialize key variables
        data = {}
        features = 14
        sheets = [0, 2]
        cls_kslr_bin = []
        cls_kslr_num = []
        classes_num = []

        #####################################################################
        #####################################################################
        # Read the feature vectors
        #####################################################################
        #####################################################################

        # Read training data from first  worksheet
        for sheet in sheets:
            # Initialize data
            data[sheet] = []
            start = False

            # Open workbook
            workbook = xlrd.open_workbook(self.filename)
            worksheet = workbook.sheet_by_index(sheet)
            for row in range(worksheet.nrows):
                # Data starts on row after the row with the heading
                # "Temperature" in the first column
                if start is False:
                    value = worksheet.row(row)[0].value
                    start = _start_processing(value)

                    # Read next row
                    continue

                # Get data
                values = []
                for column in range(0, features + 1):
                    values.append(worksheet.row(row)[column].value)

                # Apply kessler to nominal values
                for column in range(6, features + 1):
                    if values[column] == 0:
                        values[column] = -1
                    else:
                        values[column] = 1

                # Prepend a "1" to the row
                pvalues = [1]
                pvalues.extend(values)

                # Append to the list of lists
                data[sheet].append(pvalues)

                ##############################################################
                ##############################################################
                # Read the classes
                ##############################################################
                ##############################################################

                if sheet == 0:
                    # Append to binary classifier data
                    value = worksheet.row(row)[features + 1].value
                    cls_kslr_bin.append([int(value)])

                    # Get data (kessler)
                    values = [-1] * 6
                    cls_numeric = worksheet.row(row)[features + 2].value
                    values[int(cls_numeric)] = 1
                    cls_kslr_num.append(values)
                    classes_num.append(cls_numeric)

        # Return
        return (data, cls_kslr_bin, cls_kslr_num, classes_num)


def _start_processing(value):
    """Determine whether to start processing the file.

    Args:
        value: Value to check

    Returns:
        start: flag as to whether to start processing

    """
    # Initialize key variables
    start = False

    # Test start criteria
    if isinstance(value, str) is True:
        if 'temperature' in value.lower():
            start = True

    return start


def cli():
    """Read the CLI.

    Args:
        None:

    Returns:
        None

    """
    # Header for the help menu of the application
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    # Add argument
    parser.add_argument(
        '--filename',
        required=True,
        type=str,
        help='Filename to import.'
    )

    # Get the parser value
    args = parser.parse_args()

    # Return
    return args


def main():
    """Analyze data for a 2D histogram.

    Args:
        None:

    Returns:
        None:

    """
    # Initialize key values
    predicted_b = []
    predicted_n = []

    # Get data
    args = cli()

    # Start the ingest
    ingest = Ingest(args.filename)

    # Get training data and kessler classes
    training_data = ingest.training_data()
    data_to_classify = ingest.data_to_classify()
    klasses_b = ingest.klasses_binary()
    klasses_n = ingest.klasses_non_binary()
    classes_n = ingest.classes_non_binary()

    #########################################################################
    #########################################################################
    # Classifier Creation
    #########################################################################
    #########################################################################

    # Apply classifer
    classify = Linear(training_data)

    # Classifier for binary data
    classifier_b = classify.classifier(klasses_b)
    pprint(classifier_b)
    print('\n')

    # Classifier for non-binary data
    classifier_n = classify.classifier(klasses_n)
    pprint(classifier_n)
    print('\n')

    #########################################################################
    #########################################################################
    # Predictions
    #########################################################################
    #########################################################################

    print('\nPredictions')

    # Print predicted binary classes
    for vector in data_to_classify:
        next_class = classify.prediction(vector, klasses_b)
        predicted_b.append(next_class)
        print(next_class)

    print('\n')

    # Print predicted non-binary classes
    for vector in data_to_classify:
        next_class = classify.prediction(vector, klasses_n)
        predicted_n.append(next_class)
        print(next_class)

    sys.exit(0)

    #########################################################################
    #########################################################################
    # Confusion matrices
    #########################################################################
    #########################################################################

    print('\nConfusion Matrix Classification')

    # Predict classification of the original training data
    predictions_b = []
    predictions_n = []

    count = 0
    for vector in training_data:
        next_class = classify.prediction(vector, klasses_n)
        predictions_n.append(next_class)

        next_class = classify.prediction(vector, klasses_b)
        predictions_b.append(next_class)

    # Confusion matrix
    print('Confusion Matrix Calculation')

    matrix_b = confusion_matrix(
        klasses_b, np.asarray(predictions_b))
    pprint(matrix_b)

    print('\n')

    matrix_n = confusion_matrix(
        classes_n, np.asarray(predictions_n))
    pprint(matrix_n)

    print('Confusion PPV')

    # Print PPV
    ppvs = []
    for count in range(0, 6):
        column = matrix_n[:, count]
        ppv = matrix_n[count, count] / np.sum(column)
        ppvs.append((count, ppv))

    for item in sorted(ppvs, key=lambda x: x[1], reverse=True):
        print(item)

if __name__ == "__main__":
    main()
