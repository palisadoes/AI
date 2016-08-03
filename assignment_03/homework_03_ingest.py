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
    digits = [1, 2]
    none_digits = digits[:]
    none_digits.append(None)
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
            minst_array = np.asarray(minst_images[pointer]) / 256
            data.append(
                (cls, minst_array.tolist())
            )

    # Print the data shape
    print('Digits = ', digits)
    print('Data Elements = ', len(data))

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
    pc_2 = principal_components[:, 1]

    # Feed the chart
    chart_data = (principal_classes, pc_1, pc_2)
    graph = chart.ChartPC(digits, chart_data)
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
        eigenvectors = pca_object.eigenvectors()

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
                xvalues[count], components)

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
    pc_of_x = pca_object.pc_of_x(xvalue)
    output('featurevector', xvalue)
    output('zvalue', xvalue - pca_object.meanvector())
    output('principal_components', pc_of_x)
    output(
        'reconstructed_z',
        pca_object.reconstruct(xvalue, components) - pca_object.zvalues()
    )
    output('reconstructed_x', pca_object.reconstruct(xvalue, components))

    # Principal components for Gaussian calculations
    minima = np.asarray([min(pc_1), min(pc_2)])
    maxima = np.asarray([max(pc_1), max(pc_2)])
    output('minima', minima)
    output('maxima', maxima)
    print('PC 1 Min / Max', min(pc_1), max(pc_1))
    print('PC 2 Min / Max', min(pc_2), max(pc_2))

    #########################################################################
    # Calculate training accuracy
    #########################################################################
    print('Training Accuracy')

    # Calculate probabilities
    bayes_classifier = classifier2d.Bayesian(pca_object)
    histo_classifier = classifier2d.Histogram(pca_object)

    # Output class based means and covariances
    print('\nMean Vectors')
    for cls in digits:
        value = bayes_classifier.meanvector(cls=cls)
        output(('mu_%s') % (cls), value)
        print(('Class %s:') % (cls))
        pprint(value)

    print('\nCovariance')
    for cls in digits:
        value = bayes_classifier.covariance(cls=cls)
        output(('covariance_%s') % (cls), value)
        print(('Class %s:') % (cls))
        pprint(value)

    # Output Histogram data
    for cls in digits:
        output(('histogram_%s') % (cls), histo_classifier.histogram()[cls])

    # Get accuracy values
    g_accuracy = bayes_classifier.accuracy()

    print('\nGaussian Accuracy')
    for cls in none_digits:
        print(
            ('Class %s: %.2f%%') % (cls, g_accuracy[cls])
        )

    h_accuracy = histo_classifier.accuracy()

    # Print accuracy
    print('\nHistogram Accuracy')
    for cls in none_digits:
        print(
            ('Class %s: %.2f%%') % (cls, h_accuracy[cls])
        )

    # Calculate probabilities
    g_digit_class = bayes_classifier.classifier(xvalue)
    g_digit_probability = bayes_classifier.probability(xvalue)

    # Print accuracy
    print('\nGaussian Probability')
    print('Classified Value', g_digit_class)
    print(('Classified Probability: %.2f%%') % (
        g_digit_probability[g_digit_class] * 100))

    # Calculate probabilities
    h_digit_class = histo_classifier.classifier(xvalue)
    h_digit_probability = histo_classifier.probability(xvalue)

    # Print accuracy
    print('\nHistogram Probability')
    print('Classified Value', h_digit_class)
    print(('Classified Probability: %.2f%%') % (
        h_digit_probability[h_digit_class] * 100))

    #########################################################################
    # Create 3D Histogram for first 2 principal components for both classes
    #########################################################################
    print('\nCreating 3D Histogram Chart')
    histo_classifier.graph3d()


def output(label, value):
    """Output values to files.

    Args:
        None:

    Returns:
        None:

    """
    # Initialize key variables
    output_directory = '/home/peter/Downloads/UCSC/csv'
    rows = value.tolist()
    list_of_lists = any(isinstance(element, list) for element in rows)

    # Write file
    filename = ('%s/%s.csv') % (output_directory, label)
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        if list_of_lists is True:
            spamwriter.writerows(rows)
        else:
            spamwriter.writerow(rows)


if __name__ == "__main__":
    main()
