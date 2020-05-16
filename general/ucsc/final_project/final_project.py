#!/usr/bin/env python3
"""Program Ingests data for a 2D histogram."""

# Standard python imports
import argparse
import time
import csv

import numpy as np

# Import AI library
from machine import mnist
from machine import pca
from machine import classifier2d


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

    # Add argument
    parser.add_argument(
        '--components',
        required=True,
        type=int,
        help='Number of principal components.'
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
    tuple_list = []
    args = cli()
    total_components = args.components

    ##########################################################################
    # Ingest data
    ##########################################################################
    print('Ingesting Data')
    mnist_data_directory = args.mnist_data_directory
    minst_data = mnist.MNIST(mnist_data_directory)
    (minst_images, minst_labels) = minst_data.load_training()

    for next_class in range(1, 10):
        # Initialize data for loop
        data = []

        # Update what we are doing
        print('\nProcessing class', next_class)

        # Define digits
        digits = [0, next_class]

        # Get data for digits
        for pointer, cls in enumerate(minst_labels):
            if cls in digits:
                minst_array = np.asarray(minst_images[pointer]) / 256
                data.append(
                    (cls, minst_array.tolist())
                )

        # Create the PCA object for processing later
        pca_object = pca.PCA(data)

        ######################################################################
        # Instantiate PCA for next principal components
        ######################################################################
        for components in range(2, 2 + total_components):
            # Update what we are doing
            print('\tCalculating Accuracy:', components, 'components')

            # Log starting time
            ts_start = time.time()

            ##################################################################
            # Calculate training accuracy
            ##################################################################

            # Calculate probabilities
            bayes_classifier = classifier2d.Bayesian(pca_object, components)

            # Get accuracy values
            g_accuracy = bayes_classifier.accuracy()

            # Log elapsed time
            ts_stop = time.time()
            ts_elapsed = ts_stop - ts_start

            # Append results
            tuple_list.append(
                (next_class, components, g_accuracy[next_class], ts_elapsed)
            )

    # All done export to file
    filename = (
        '/home/peter/Downloads/UCSC_project-%s.csv') % (int(time.time()))
    with open(filename, 'w') as f_handle:
        writer = csv.writer(f_handle, delimiter=',', lineterminator='\n')
        writer.writerows(tuple_list)


if __name__ == "__main__":
    main()
