#!/usr/bin/env python3
"""Program creates histograms."""

import argparse
import textwrap
import sys

import xlrd

# Import AI library
from machine.histogram1d import Histogram1D


def cli():
    """Return all the CLI options.

    Args:
        None

    Returns:
        args: Namespace() containing all of our CLI arguments as objects
            - filename: Path to the configuration file

    """
    # Header for the help menu of the application
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    # Process config_file
    parser.add_argument(
        '--datafile',
        required=True,
        default=None,
        type=str,
        help=textwrap.fill('The data file to use.')
    )

    # Return the CLI arguments
    args = parser.parse_args()

    # Return our parsed CLI arguments
    return args


def getdata(filename, maxrows=None):
    """Function for intializing the class."""
    # Initialize key variables
    data = []
    entries = 0

    # Read spreadsheet
    workbook = xlrd.open_workbook(filename)
    worksheet = workbook.sheet_by_index(0)
    for row in range(worksheet.nrows):
        # Get data
        feet = worksheet.row(row)[0].value
        inches = worksheet.row(row)[1].value
        gender = worksheet.row(row)[2].value

        # Skip header, append data
        if 'gender' not in gender.lower():
            # Get minimum and maximum values of heights
            height = int(feet * 12 + inches)

            # Update data with heights
            data.append(
                (gender.lower(), height)
            )

            # Conditional break
            entries = entries + 1
            if maxrows is not None:
                if entries >= maxrows:
                    break

    # Return
    return data


def main():
    """Process data."""
    # Initialize key variables
    feature = 'Height'

    # Process the CLI
    args = cli()

    # We are only doing histogram stuff
    if args.mode != 'histogram':
        sys.exit(0)

    # Get data
    data_list = getdata(args.datafile)
    data = Histogram1D(data_list)
    data.graph(feature)
    data.table(feature)
    data.parameters()

    print('\n')
    data_list = getdata(args.datafile, maxrows=200)
    data = Histogram1D(data_list)
    data.graph(feature)
    data.table(feature)
    data.parameters()

if __name__ == "__main__":
    main()
