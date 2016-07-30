#!/usr/bin/env python3
"""Program Ingests data for a 2D histogram."""

# Standard python imports
import argparse
from pprint import pprint
import time
import sys
import csv

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
    digits = [1, 9]
    maximages = 5
    components = 2
    data = []

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

    #########################################################################
    # Do scatter plot
    # http://peekaboo-vision.blogspot.com/2012/12/another-look-at-mnist.html
    #########################################################################
    print('Creating Scatter Plot')

    # Retrieve principal components and their associated classes
    (principal_classes,
     principal_components) = pca_object.principal_components(
         components=components)

    # Save principal components for later use
    pc_1 = principal_components[:, 0]
    pc_2 = principal_components[:, 0]

    # Feed the chart
    chart_data = (principal_classes, pc_1, pc_2)
    graph = chart.Chart(digits, chart_data)
    graph.graph()

    #########################################################################
    # View eigenvectors as images
    #########################################################################
    print('Creating Covariance Matrix Image')
    for cls in digits:
        # Create image
        covariance = pca_object.covariance(cls)
        pca.image_by_list(covariance, ('%s-covariance') % (cls))

    # Create image for all classes
    covariance = pca_object.covariance(None)
    pca.image_by_list(covariance, ('%s-covariance') % (None))

    #########################################################################
    # View eigenvectors as images
    #########################################################################
    print('Creating Eigenvector Based Images')
    for cls in digits:
        eigenvectors = pca_object.eigenvectors(cls=None, sort=True)

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
            image = pca_object.reconstruct(
                xvalues[count], None, components)

            # Create image from principal component
            pca.image_by_list(xvalues[count], ('%s-%s-orig') % (cls, count))
            pca.image_by_list(image, ('%s-%s-reconstruct') % (cls, count))

    #########################################################################
    # Output metadata for whole dataset
    #########################################################################
    print('Data output - All Classes')
    output('meanvector', pca_object.meanvector())
    output('eigenvector_1', pca_object.eigenvectors()[0])
    output('eigenvector_2', pca_object.eigenvectors()[1])

    tcls = digits[0]
    print(('Data output - Class Digit [%s]') % (tcls))

    # XZCVPR values
    xvalue = pca_object.xvalues(tcls)[0]
    output('featurevector', xvalue)
    output('zvalue', pca_object.zvalues(tcls))
    output('principal_components', pca_object.principal_components(tcls)[1])
    output(
        'reconstructed_z',
        pca_object.reconstruct(
            xvalue, tcls, components) - pca_object.zvalues(tcls)
    )
    output('reconstructed_x', pca_object.reconstruct(xvalue, tcls, components))

    # Principal components for Gaussian calculations
    minima = np.asarray([min(pc_1), min(pc_2)])
    maxima = np.asarray([max(pc_1), max(pc_2)])
    output('minima', minima)
    output('maxima', maxima)

    #########################################################################
    # Calculate training accuracy
    #########################################################################
    print('Training Accuracy')

    # Loop through data to create chartable lists by class
    new_data = []
    for (col, ), cls in np.ndenumerate(principal_classes):
        new_data.append(
            (cls, pc_1[col])
        )
        new_data.append(
            (cls, pc_2[col])
        )

    # Instantiate the object
    pca_new = pca.PCA(new_data)

    # Output class based means and covariances
    for cls in digits:
        output(('mu_%s') % (cls), pca_new.meanvector(cls=cls))
        output(('covariance_%s') % (cls), pca_new.covariance(cls=cls))

    # Calculate probabilities
    probability = pca.Probability2D(pca_new)

    # Output Histogram data
    for cls in digits:
        output(('histogram_%s') % (cls), probability.histogram()[cls])

    # Get accuracy values
    g_accuracy = probability.gaussian_accuracy()
    h_accuracy = probability.histogram_accuracy()

    # Print accuracy
    print('\nHistogram Accuracy')
    for cls in digits:
        print(
            ('Class %s: %s%%') % (cls, h_accuracy[cls])
        )

    print('\nGaussian Accuracy')
    for cls in digits:
        print(
            ('Class %s: %s%%') % (cls, g_accuracy[cls])
        )


def output(label, value):
    """Output values to files.

    Args:
        None:

    Returns:
        None:

    """
    # Initialize key variables
    output_directory = '/home/peter/Downloads/UCSC/csv'
    row = [label].extend(value)
    print(row)

    # Write file
    filename = ('%s/%s.csv') % (output_directory, label)
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(
            csvfile, delimiter='\t',
            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(row)


if __name__ == "__main__":
    main()
