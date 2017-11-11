#!/usr/bin/env python3
"""Script to demonstrate  kmeans machine learning."""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')
from sklearn import preprocessing
import pandas as pd
import numpy as np


def main():
    """Main Function.

    Display kmeans data

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
    dataframe = pd.read_excel('data/titanic.xls')
    print(dataframe.head())

    # Drop unnecessary columns and convert values to integer
    dataframe.drop(['body','name'], 1, inplace=True)
    dataframe.convert_objects(convert_numeric=True)
    dataframe.fillna(0, inplace=True)

    # Convert non-numerical data in columns to an equivalent integer value
    dataframe = handle_non_numerical_data(dataframe)
    print(dataframe.head())

    x_ary = np.array(dataframe.drop(['survived'], 1).astype(float))
    y_ary = np.array(dataframe['survived'])

    clf = KMeans(n_clusters=2)
    clf.fit(x_ary)


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
