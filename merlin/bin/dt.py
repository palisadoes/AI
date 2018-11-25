#!/usr/bin/env python3
"""Script to forecast data using RNN AI using GRU feedback."""

# Standard imports
import argparse
import sys

# PIP3 imports.
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
    parser.add_argument(
        '-s', '--steps', help='Number of steps into the future to predict.',
        type=int, default=0)
    parser.add_argument(
        '--actuals',
        help=(
            'Use actual values in creating the decision tree vectors. '
            'Default=False.'),
        action='store_true')
    args = parser.parse_args()
    filename = args.filename
    actuals = args.actuals
    steps = args.steps

    # Get the data for processing
    data = database.DataDT(filename, actuals=actuals, steps=steps)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data.vectors(),
        data.classes(),
        test_size=0.2,
        random_state=42,
    )

    '''for item in data.vectors():
        print(item)'''

    tree = DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)

    print(
        '> The Decison Tree prediction accuracy is: {:.2f}%'.format(
            tree.score(x_test, y_test) * 100))

    ################################################

    num_trees = 100
    max_features = 3
    kfold = model_selection.KFold(n_splits=10)
    model = RandomForestClassifier(
        n_estimators=num_trees, max_features=max_features)
    results = model_selection.cross_val_score(
        model, data.vectors(), data.classes().flatten(), cv=kfold)

    print(
        '> The Random Forest prediction accuracy is: {:.2f}%'.format(
            results.mean() * 100))


if __name__ == "__main__":
    main()
