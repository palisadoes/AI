#!/usr/bin/env python3
"""Program Ingests data for a 2D histogram."""

# Standard python imports
import argparse
from pprint import pprint
import time
import sys

from sklearn.decomposition import PCA as PCX

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
    digits = [5, 7]
    maximages = 5
    components = 2
    data = []

    """
    #########################################################################
    # Test
    #########################################################################

    data = []
    testclasses = ['1_class', '2_class']
    tcls = testclasses[0]
    xarray = np.array(
        [[-3, 5, 0],
         [-3, 4, -1],
         [-4, 0, -1],
         [-1, -3, -3]])
    xarray_list = np.ndarray.tolist(xarray)
    for cls in sorted(testclasses):
        for row in xarray_list:
            data.append(
                (cls, row)
            )

    # Instantiate PCA
    pct = pca.PCA(data)

    print('X')
    pprint(pct.xvalues(tcls))
    print('Z')
    pprint(pct.zvalues(tcls))
    print('C')
    pprint(pct.covariance(tcls))
    print('V')
    pprint(pct.eigenvectors(tcls))
    print('P')
    pprint(pct.principal_components(tcls))
    print('R')
    pprint(pct.reconstruct(xarray_list, tcls, 3))
    """

    #########################################################################
    # Ingest data
    #########################################################################
    print('Ingesting Data')
    data = []
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

    #########################################################################
    # Instantiate PCA
    #########################################################################
    print('Analyzing Data')
    pca_object = pca.PCA(data)

    """
    values = {}
    testdata = []
    pcx_r = {}
    pcx = PCX(n_components=2)
    for cls in digits:
        values[cls] = pca_object.xvalues(cls)
        pcx_r[cls] = pcx.fit_transform(values[cls])
        print(type(pcx_r[cls]))
        print(pcx_r[cls].shape)
        testdata.append(
            (cls,
             pcx_r[cls][:, 0],
             pcx_r[cls][:, 1])
        )
    graph = chart.Chart(testdata)
    graph.graph()
    sys.exit()
    """

    #########################################################################
    # Do scatter plot
    #########################################################################
    print('Creating Scatter Plot')
    data = []
    for cls in digits:
        principal_components = pca_object.principal_components(
            cls, components=components)
        data.append(
            (cls,
             principal_components[:, 0],
             principal_components[:, 1])
            # principal_components[:, 0],
            # principal_components[:, 1])
        )
    graph = chart.Chart(data)
    graph.graph()

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
            image = pca_object.reconstruct(xvalues[count], cls, components)

            # Create image from principal component
            pca.image_by_list(xvalues[count], ('%s-%s-orig') % (cls, count))
            pca.image_by_list(image, ('%s-%s-reconstruct') % (cls, count))

    #########################################################################
    # Calculate training accuracy
    #########################################################################
    probability = pca.Probability2D(pca_object)
    g_accuracy = probability.gaussian_accuracy()
    h_accuracy = probability.histogram_accuracy()

    # Print accuracy
    for cls in digits:
        print('Histogram Accuracy')
        print(
            ('Class %s: %s%%') % (cls, h_accuracy[cls])
        )

    for cls in digits:
        print('Gaussian Accuracy')
        print(
            ('Class %s: %s%%') % (cls, g_accuracy[cls])
        )


if __name__ == "__main__":
    main()
