#!/usr/bin/env python3
"""Program Ingests data for a 2D histogram."""

# Standard python imports
import argparse
from pprint import pprint

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
        (data, _, _) = self.data
        return data[0]

    def data_to_classify(self):
        """Method to obtain training data.

        Args:
            None

        Returns:
            None

        """
        (data, _, _) = self.data
        return data[2]

    def binary(self):
        """Method to obtain training data.

        Args:
            None

        Returns:
            None

        """
        (_, data, _) = self.data
        return data

    def non_binary(self):
        """Method to obtain training data.

        Args:
            None

        Returns:
            None

        """
        (_, _, data) = self.data
        return data

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
        binary = []
        non_binary = []

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

                # print('_data', sheet, pvalues, values, '\n')

                ##############################################################
                ##############################################################
                # Read the classes
                ##############################################################
                ##############################################################

                if sheet == 0:
                    # Append to binary classifier data
                    value = worksheet.row(row)[features + 1].value
                    binary.append([int(value)])

                    # Get data
                    values = [-1] * 6
                    non_binary_column = worksheet.row(row)[features + 2].value
                    values[int(non_binary_column)] = 1
                    non_binary.append(values)

                # Append to the list of lists
                data[sheet].append(pvalues)

        # Return
        return (data, [binary], non_binary)


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
    # Get data
    args = cli()

    # Start the ingest
    ingest = Ingest(args.filename)

    # Get training data and kessler classes
    training_data = ingest.training_data()
    binary_classes = ingest.binary()
    non_binary_classes = ingest.non_binary()

    # Apply classifer
    classify = Linear(training_data)
    pprint(classify.classifier(binary_classes))
    pprint(classify.classifier(non_binary_classes))

if __name__ == "__main__":
    main()
