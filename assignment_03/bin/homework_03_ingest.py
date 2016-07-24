#!/usr/bin/env python3
"""Program Ingests data for a 2D histogram."""

# Standard python imports
import argparse
from pprint import pprint
import time

# Non standard python imports
import numpy as np

# Import AI library
from machine import mnist
from machine import pca
from machine import chart


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
    digits = [5, 6]
    maximages = 5
    components = 2
    data = []

    #########################################################################
    # Ingest data
    #########################################################################
    print('Ingesting Data')
    args = cli()
    mnist_data_directory = args.mnist_data_directory
    minst_data = mnist.MNIST(mnist_data_directory)
    (minst_images, minst_labels) = minst_data.load_training()

    # Get data for digits
    for pointer, cls in enumerate(minst_labels):
        if cls in digits:
            data.append(
                (cls, minst_images[pointer])
            )

    # Instantiate PCA
    pca_object = pca.PCA(data)

    #########################################################################
    # View eigenvectors as images
    #########################################################################
    print('Creating Covariance Matrix Image')
    for cls in digits:
        # Create image
        covariance = pca_object.covariance(cls)
        pca.image_by_list(covariance, ('%s-covariance') % (cls))

    #########################################################################
    # View eigenvectors as images
    #########################################################################
    print('Creating Eigenvector Based Images')
    for cls in digits:
        eigenvectors = pca_object.eigenvectors(cls, sort=True)

        count = 0
        for eigenvector in eigenvectors:
            # Sleep the image files get a chance to be written to disk OK
            time.sleep(1)

            # Create image from eigenvector
            pca.image_by_list(eigenvector, ('%s-%s-eigen') % (cls, count))
            count += 1
            if count == maximages:
                break

    #########################################################################
    # Do scatter plot
    #########################################################################
    print('Creating Scatter Plot')
    data = []
    for cls in digits:
        """
        principal_components = pca_object.principal_components(
            cls, components=components)
        data.append(
            (cls,
             principal_components[:, 0],
             principal_components[:, 1])
            # principal_components[0],
            # principal_components[1])
        )
        """
        eigens = pca_object.eigenvectors(cls, components=components)
        data.append(
            (cls,
             eigens[0],
             eigens[1])
        )
    graph = chart.Chart(data)
    graph.graph()

    #########################################################################
    # Reconstruct first five images from principal components
    #########################################################################
    print('Recreating Images')
    for cls in digits:
        # Stage required variables
        xvalues = pca_object.xvalues(cls)

        # Create 2 x N eigen vector array
        for count in range(0, maximages):
            # Sleep so image files get a chance to be written to disk OK
            time.sleep(1)

            # Reconstruct image from principal component
            imagery = pca.PCAx(xvalues[count], components, cls, pca_object)
            image = imagery.reconstruct()

            # Create image from principal component
            pca.image_by_list(xvalues[count], ('%s-%s-orig') % (cls, count))
            pca.image_by_list(image, ('%s-%s-reconstruct') % (cls, count))


if __name__ == "__main__":
    main()
