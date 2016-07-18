#!/usr/bin/env python3
"""Program Ingests data for a 2D histogram."""

# Standard python imports
import argparse
from pprint import pprint
from collections import defaultdict

# Import AI library
from machine import mnist
from machine import pca


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
        '--mnist_data_directory',
        required=True,
        type=str,
        help='Directory with the MNIST data.'
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
    # Initialize key variables
    digits = [1, 0]
    data = []

    # Ingest data
    args = cli()
    mnist_data_directory = args.mnist_data_directory
    minst_data = mnist.MNIST(mnist_data_directory)
    (minst_images, minst_labels) = minst_data.load_training()

    # Get data for digits
    for pointer, value in enumerate(minst_labels):
        if value in digits:
            data.append(
                (value, minst_images[pointer])
            )

    # Instantiate PCA
    pca_object = pca.PCA2d(data)
    # pca_object.image(1, 500)
    # print(pca_object.zvalues(1)[0])
    # pca_object.zvalues(1)
    matrix = pca_object.covariance(0)
    pca.image_by_list(matrix)


if __name__ == "__main__":
    main()
