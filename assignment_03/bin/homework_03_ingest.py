#!/usr/bin/env python3
"""Program Ingests data for a 2D histogram."""

# Standard python imports
import argparse
import time
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

    """
    values = pca_object.eigen_values_vectors(1)
    pprint(values[0])
    print('\n')
    pprint(values[1])
    print('\n')
    print(values[0].shape)
    print(values[1].shape)
    print('\n')
    print(values[1][400])
    print('\n')
    print(values[1][400][0])
    print(values[1][400][1])
    print(values[1][400][2])
    print('\n')
    eigens = pca_object.eigen_vectors(1)
    pprint(eigens[400])
    print('\n')
    checker = pca_object.eigen_vector_check(1)
    pprint(checker)
    print(pca_object.zvalues(1)[0])
    pca_object.zvalues(1)
    print('Quick', int(time.time()))
    matrix = pca_object.covariance_manual(0)
    pca.image_by_list(matrix)
    print('Slow', int(time.time()))
    matrix = pca_object.covariance(0)
    pca.image_by_list(matrix)
    print('Done', int(time.time()))

    eigens = pca_object.eigen_vectors(1)
    count = 0
    for eigen in eigens:
        pprint(eigen)
        pca_object.image_by_vector(eigen)
        time.sleep(1.1)
        if count == 4:
            break
        count += 1
    """
    covariance = pca_object.covariance(0)
    pprint(covariance[400])
    pprint(covariance[0])
    print('------------------')
    eigens = pca_object.eigen_vectors(0)
    pprint(eigens[0])
    pprint(eigens[1])
    pprint(eigens[2])
    pprint(eigens[3])
    # pca_object.image_by_vector(eigens[400])


if __name__ == "__main__":
    main()
