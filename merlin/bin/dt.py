#!/usr/bin/env python3
"""Script to forecast data using RNN AI using GRU feedback."""

# Standard imports
import argparse
import sys
import time

# PIP3 imports.
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

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
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data.vectors(),
        data.classes(),
        test_size=0.3,
        random_state=42,
    )

    tree = DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)

    print(
        '> The Decison Tree prediction accuracy is: {:.2f}%'.format(
            tree.score(x_test, y_test) * 100))

    ################################################

    '''print(y_train.flatten())
    print(y_train.shape)
    sys.exit()'''

    num_trees = 100
    max_features = 3
    kfold = model_selection.KFold(n_splits=10)
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    results = model_selection.cross_val_score(model, data.vectors(), data.classes().flatten(), cv=kfold)

    print(
        '> The Random Forest prediction accuracy is: {:.2f}%'.format(
            results.mean() * 100))


if __name__ == "__main__":
    main()
