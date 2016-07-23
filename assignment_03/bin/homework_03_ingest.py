#!/usr/bin/env python3
"""Program Ingests data for a 2D histogram."""

# Standard python imports
import argparse
from pprint import pprint
from collections import defaultdict
from random import randint

# Non standard python imports
import numpy as np

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
    test_key = 6
    digits = [5, 6]
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
    pca_object = pca.PCA(data)
    maximages = 5

    print('Eigen')
    eigenvectors = pca_object.eigenvectors(test_key)

    # Visually confirm that the list is correctly
    # sorted by decreasing eigenvalues
    print('Eigenvalues in descending order:')
    count = 0
    for eigenvector in eigenvectors:
        # real_vector = np.real(eigenvector)
        # pprint(real_vector)
        pca.image_by_list(eigenvector)
        count += 1
        if count == maximages:
            break

    # Try recreating images
    number_of_components = 2
    principal_components = pca_object.principal_components(test_key)
    print(principal_components.shape)
    p1p2 = principal_components[: 1:][:, : 2]
    v1v2 = eigenvectors[: 2:, ]
    image_vectors = np.dot(p1p2, v1v2)
    print(image_vectors.shape)
    pprint(image_vectors)
    pca.image_by_list(image_vectors)

    # Test
    data = []
    for digit in digits:
        for row in range(1, 6):
            data.append(
                (digit, (
                    row,
                    randint(0, 30),
                    randint(0, 30),
                    randint(0, 30),
                    randint(0, 30)))
            )
    pprint(data)

    # Initialize again
    pca_object = pca.PCA(data)

    print('\nX Values')
    pprint(pca_object.xvalues(test_key))

    print('\nMean Vector')
    pprint(pca_object.meanvector(test_key))

    print('\nZ Values')
    pprint(pca_object.zvalues(test_key))

    print('\nCovariance Manual')
    pprint(pca_object.covariance_manual(test_key))

    print('\nCovariance Builtin')
    covariance = pca_object.covariance(test_key)
    pprint(covariance)

    print('\nEigen Values & Vectors')
    pprint(pca_object._eigen_values_vectors(test_key))

    print('\nEigen Vectors')
    eigen = pca_object.eigenvectors(test_key)
    pprint(eigen)
    total = 0
    for column in range(0, len(eigen[0])):
        total = eigen[0, column] * eigen[4, column]
    print(total)

    print('\nEigen Vector Check')
    pprint(pca_object.eigen_vector_check(test_key))


if __name__ == "__main__":
    main()
