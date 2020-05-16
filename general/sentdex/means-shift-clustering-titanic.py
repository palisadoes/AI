#!/usr/bin/env python3
"""Script to demonstrate  MeanShift machine learning."""

from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.cluster import MeanShift
style.use('ggplot')


def main():
    """Main Function.

    Display MeanShift data

    Data retrieved from: https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls

    Format:

        Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
        survival Survival (0 = No; 1 = Yes)
        name Name
        sex Sex
        age Age
        sibsp Number of Siblings/Spouses Aboard
        parch Number of Parents/Children Aboard
        ticket Ticket Number
        fare Passenger Fare (British pound)
        cabin Cabin
        embarked Port of Embarkation (
            C = Cherbourg; Q = Queenstown; S = Southampton)
        boat Lifeboat
        body Body Identification Number
        home.dest Home/Destination

    """
    # Import data
    dataframe = pd.read_excel('data/titanic.xls')

    # Create a copy of the original dataframe
    # This is done solely for analysis
    original_dataframe = pd.DataFrame.copy(dataframe)

    # Drop unnecessary columns and convert values to integer
    dataframe.drop(['body', 'name'], 1, inplace=True)
    dataframe.convert_objects(convert_numeric=True)
    dataframe.fillna(0, inplace=True)

    # Convert non-numerical data in columns to an equivalent integer value
    dataframe = handle_non_numerical_data(dataframe)

    # Strip the classes from the dataframe
    x_ary = np.array(dataframe.drop(['survived'], 1).astype(float))
    y_ary = np.array(dataframe['survived'])

    # Do preprocessing on x_ary to make predictions more accurate
    x_ary = preprocessing.scale(x_ary)

    # Classify
    clf = MeanShift()
    clf.fit(x_ary)

    # Get data labels and the cluster centers from the classifier
    labels = clf.labels_
    cluster_centers = clf.cluster_centers_

    # Add a new column to our original dataframe and iterate through the
    # labels and populate the labels to the empty column
    original_dataframe['cluster_group'] = np.nan
    for item in range(len(x_ary)):
        original_dataframe['cluster_group'].iloc[item] = labels[item]

    # Check the survival rates for each of the groups we happen to find
    n_clusters_ = len(np.unique(labels))
    survival_rates = {}
    for item in range(n_clusters_):
        # Create a temporary dataframe for the Nth cluster (class) found
        temp_dataframe = original_dataframe[
            (original_dataframe['cluster_group'] == float(item))]

        # Lets get the survival rate for this cluster / class
        survival_cluster = temp_dataframe[(temp_dataframe['survived'] == 1)]
        survival_rate = len(survival_cluster) / len(temp_dataframe)
        survival_rates[item] = survival_rate

    # Print survival rates
    print(survival_rates, '\n')

    """
    Review the data for each class.
    You'll notice that the clustering appears to mimic the class of the
    ticket that was purchsased by the passenger.
    """
    for item in sorted(survival_rates.keys()):
        print(
            original_dataframe[
                (original_dataframe['cluster_group'] == item)].describe())
        print('\n==================================\n')

    # What was the survival_rate for first class passengers in cluster 0?
    cluster_0 = (original_dataframe[
        (original_dataframe['cluster_group'] == 0)])
    cluster_0_fc = (cluster_0[(cluster_0['pclass'] == 1)])
    print(cluster_0_fc.describe())


def handle_non_numerical_data(dataframe):
    """Remove non-numerical data.

    Args:
        dataframe: Dataframe

    Returns:
        dataframe: Modified dataframe

    """
    # Initialize key variables
    columns = dataframe.columns.values

    # Iterate over each column in the dataframe
    for column in columns:
        # Create an empty dictionary for this column
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        # Convert non-numerical values
        if (dataframe[column].dtype != np.int64) and (
                dataframe[column].dtype != np.float64):

            # Convert data to list
            column_contents = dataframe[column].values.tolist()
            unique_elements = set(column_contents)

            count = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = count
                    count += 1

            dataframe[column] = list(
                map(convert_to_int, dataframe[column]))

    return dataframe


if __name__ == "__main__":
    main()
