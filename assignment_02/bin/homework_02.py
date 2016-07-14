#!/usr/bin/env python3
"""Program Ingests data for a 2D histogram."""

# Standard python imports
import argparse
from pprint import pprint
from collections import defaultdict

# Non-standard python imports
import xlrd

# Custom libraries
from machine.histogram import Histogram1D


def ingest(filename):
    """Read the ingest excel file.

    Args:
        filename: Filename

    Returns:
        data: List of tuples of data

    """
    # Initialize key variables
    data = defaultdict(lambda: defaultdict(dict))
    labels = None

    # Read spreadsheet
    workbook = xlrd.open_workbook(filename)
    worksheet = workbook.sheet_by_index(0)
    for row in range(worksheet.nrows):
        # Get data
        sex = worksheet.row(row)[0].value
        sex = sex.replace('\'', '')
        height = worksheet.row(row)[1].value
        handspan = worksheet.row(row)[2].value

        # Populate data
        if 'height' not in data:
            data['height'] = []
            data['handspan'] = []

        # Skip header, append data
        if 'sex' not in sex.lower():
            # Update data with heights
            data['height'].append(
                (height, sex.lower())
            )
            data['handspan'].append(
                (handspan, sex.lower())
            )

    # Return
    return data


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

    # CLI argument for stopping
    parser.add_argument(
        '--filename',
        required=True,
        type=str,
        help='Filename to read.'
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
    # Ingest data
    args = cli()
    data = ingest(args.filename)

    # View histogram data
    for dimension in sorted(data.keys()):
        histogram = Histogram1D(data[dimension], dimension)
        histogram.graph()
        histogram.table()
        histogram.parameters()
        print('\n')



if __name__ == "__main__":
    main()
