#!/usr/bin/env python3
"""Script to forecast data using RNN AI using GRU feedback."""

# Standard imports
import argparse
import sys
import time

# PIP3 imports.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Merlin imports
from merlin import database


def main():
    """Generate forecasts.

    Display data prediction from tensorflow model

    """
    # Get CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filename', help='Name of CSV file to read.',
        type=str, required=True)
    args = parser.parse_args()
    filename = args.filename

    # Get the data for processing
    data = database.DataDT(filename)
    x_train, x_test, y_train, y_test = train_test_split(
        data.vectors(),
        data.classes(),
        test_size=0.3,
        random_state=42,
    )

    tree = DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)

    print(
        '> The prediction accuracy is: {:.2f}%'.format(
            tree.score(x_test, y_test) * 100))


if __name__ == "__main__":
    main()
