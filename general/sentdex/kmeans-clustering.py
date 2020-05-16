#!/usr/bin/env python3
"""Script to demonstrate  kmeans machine learning."""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')


def main():
    """Main Function.

    Display kmeans data

    """
    # Initialize key variables
    x_ary = np.array([
        [1, 2],
        [1.5, 1.8],
        [5, 8],
        [8, 8],
        [1, 0.6],
        [9, 11]])

    # Plot data
    """
    plt.scatter(x_ary[:, 0], x_ary[:, 1], s=150, linewidths=5, zorder=10)
    plt.show()
    """

    # Create the classifier
    clf = KMeans(n_clusters=2)
    clf.fit(x_ary)

    # Get the centroids and their labels
    centroids = clf.cluster_centers_
    labels = clf.labels_

    print(centroids, labels, len(x_ary))

    # Plot the array with its centroids
    colors = ['g.', 'r.', 'c.', 'y.']
    for i in range(len(x_ary)):
        plt.plot(x_ary[i][0], x_ary[i][1], colors[labels[i]], markersize=10)
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        marker="x", s=150, linewidths=5, zorder=10)
    plt.show()


if __name__ == "__main__":
    main()
