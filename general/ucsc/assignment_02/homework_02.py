#!/usr/bin/env python3
"""Program Ingests data for a 2D histogram."""

# Standard python imports
import argparse
from pprint import pprint
from collections import defaultdict

# Non-standard python imports
import xlrd

# Custom libraries
from machine.histogram2d import Histogram2D


def ingest(filename):
    """Read the ingest excel file.

    Args:
        filename: Filename

    Returns:
        data: List of tuples of data

    """
    # Initialize key variables
    # data = defaultdict(lambda: defaultdict(dict))
    data = []
    labels = ()

    # Read spreadsheet
    workbook = xlrd.open_workbook(filename)
    worksheet = workbook.sheet_by_index(0)
    for row in range(worksheet.nrows):
        # Get data
        sex = worksheet.row(row)[0].value
        sex = sex.replace('\'', '')
        height = worksheet.row(row)[1].value
        handspan = worksheet.row(row)[2].value

        # Skip header, append data
        if 'sex' not in sex.lower():
            # Update data with heights
            data.append(
                (sex.lower(), height, handspan)
            )
        else:
            labels = (
                height.replace('\'', ''),
                handspan.replace('\'', ''))

    # Return
    return (data, labels)


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
    (data, labels) = ingest(args.filename)

    # View histogram data
    histogram = Histogram2D(data, labels)
    histogram.graph3d()


if __name__ == "__main__":
    main()
