#!/usr/bin/env python3
"""Program Ingests data for a 2D histogram."""

# Standard python imports
import sys
import argparse
import time
from pprint import pprint
from collections import defaultdict
from random import randint
import operator

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
    pca_object = pca.PCA2d(data)
    maximages = 10

    """
    for image in range(0, maximages):
        pca_object.image(test_key, image)
        time.sleep(2)
    """

    print('Covariance')
    matrix = pca_object.covariance(test_key)

    print('Eigen')
    (eigenvalues, eigenvectors) = pca_object.eigen_values_vectors(test_key)

    """
    print('Eigen Images')
    maximages = 10
    for image in range(0, len(eigenvectors)):
        print (image)
        pca.image_by_list(eigenvectors[image])
    """

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(
        eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=operator.itemgetter(0))
    eig_pairs.reverse()

    # Visually confirm that the list is correctly
    # sorted by decreasing eigenvalues
    print('Eigenvalues in descending order:')
    count = 0
    for _, eigenvector, _ in eig_pairs:
        # real_vector = np.real(eigenvector)
        # pprint(real_vector)
        pca.image_by_list(eigenvector)
        count += 1
        if count == 5:
            break

    # Try recreating images
    number_of_components = 2
    principal_components = pca_object.principal_components(test_key)
    p1p2 = principal_components[: 1:][:, : 2]
    v1v2 = eigenvectors[: 2:, ]
    image_vectors = np.dot(p1p2, v1v2)
    print(image_vectors.shape)
    pprint(image_vectors)
    pca.image_by_list(image_vectors)

    """
    print('Eigen')
    print(vectors[0])
    print('\n')
    print(pca_object.xvalues(test_key)[0])
    print('\n')
    print(pca_object.covariance(test_key)[0])
    sys.exit(0)
    """

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

    pca_object.image(test_key, 500)
    eigens = pca_object.eigen_vectors(test_key)
    #print(pca_object.eigen_values_vectors(test_key))
    count = 0
    for eigen in eigens:
        # pprint(eigen)
        pca_object.image_by_vector(eigen)
        time.sleep(1.1)
        if count == 4:
            break
        count += 1

    matrix = pca_object.covariance(test_key)
    pca.image_by_list(matrix)

    covariance = pca_object.covariance(test_key)
    pprint(covariance[400])
    pprint(covariance[0])
    print('------------------')
    eigens = pca_object.eigen_vectors(test_key)
    pprint(eigens[0])
    pprint(eigens[1])
    pprint(eigens[2])
    pprint(eigens[3])
    # pca_object.image_by_vector(eigens[400])
    """

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
    pca_object = pca.PCA2d(data)

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
    pprint(pca_object.eigen_values_vectors(test_key))

    print('\nEigen Vectors')
    eigen = pca_object.eigen_vectors(test_key)
    pprint(eigen)
    total = 0
    for column in range(0, len(eigen[0])):
        total = eigen[0, column] * eigen[4, column]
    print(total)

    print('\nEigen Vector Check')
    pprint(pca_object.eigen_vector_check(test_key))

    """
    (rows, columns) = eigen.shape
    matrix = np.zeros(shape=(1, columns))
    for (row, column), _ in np.ndenumerate(matrix):
        summation = 0
        for ptr_col in range(0, columns):
            summation = summation + (
                eigen[row, ptr_col] * covariance[ptr_col, column])
        matrix[row, column] = summation

    check = np.zeros(shape=(1, rows))
    for row in check:
        summation = 0
        for column in check.T:
            summation = summation + (
                matrix[row, column] * matrix[row, column])
        check[row] = math.sqrt(summation)

    zeros = np.zeros(shape=(1, rows))
    for (row, column), _ in np.ndenumerate(matrix):
        summation = 0
        for column in
    """

if __name__ == "__main__":
    main()
