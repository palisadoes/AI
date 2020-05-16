#!/usr/bin/env python3
"""Script to demonstrate  MeanShift machine learning."""

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")


def main():
    """Main Function.

    Display MeanShift data

    """
    # Create three dimensional centers for the creation of sample data
    # (These are not the MeanShift centers)
    centers_to_create_sample_data = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]
    x_ary, _ = make_blobs(
        n_samples=100, centers=centers_to_create_sample_data, cluster_std=1.5)

    # Create MeanShift classifier
    classifier = MeanShift()
    classifier.fit(x_ary)
    labels = classifier.labels_
    cluster_centers = classifier.cluster_centers_

    # Show clusters
    print(cluster_centers)
    n_clusters_ = len(np.unique(labels))
    print("Number of estimated clusters:", n_clusters_)

    # Set stage for the creation of charts
    colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y', 'm']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plot of data
    for item in range(len(x_ary)):
        ax.scatter(
            x_ary[item][0], x_ary[item][1], x_ary[item][2],
            c=colors[labels[item]], marker='o')

    # Create scatter plot of centers
    ax.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
        marker='x', color='k', s=150, linewidths = 5, zorder=10)

    # Display plot
    plt.show()


if __name__ == "__main__":
    main()
