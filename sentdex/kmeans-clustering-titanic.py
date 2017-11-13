#!/usr/bin/env python3
"""Script to demonstrate  kmeans machine learning."""

from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.cluster import KMeans
style.use('ggplot')


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
    dataframe.drop(['body', 'name'], 1, inplace=True)
    dataframe.convert_objects(convert_numeric=True)
    dataframe.fillna(0, inplace=True)

    # Convert non-numerical data in columns to an equivalent integer value
    dataframe = handle_non_numerical_data(dataframe)
    print(dataframe.head())

    # Strip the classes from the dataframe
    x_ary = np.array(dataframe.drop(['survived'], 1).astype(float))
    y_ary = np.array(dataframe['survived'])

    # Do preprocessing on x_ary to make predictions more accurate
    x_ary = preprocessing.scale(x_ary)

    # Classify with two kmeans clusters
    clf = KMeans(n_clusters=2)
    clf.fit(x_ary)

    correct = 0
    for item in range(len(x_ary)):
        passenger_vector = np.array(x_ary[item].astype(float))
        passenger_vector = passenger_vector.reshape(-1, len(passenger_vector))
        prediction = clf.predict(passenger_vector)

        # Prediction is an array, matching the array of values input into the
        # classifier
        if prediction[0] == y_ary[item]:
            correct += 1

    """
    NOTE:

    https://stackoverflow.com/questions/37842165/sklearn-calculating-accuracy-score-of-k-means-on-the-test-data-set

    You should remember that k-means is not a classification tool, thus
    analyzing accuracy is not a very good idea. You can do this, but this is
    not what k-means is for. It is supposed to find a grouping of data which
    maximizes between-clusters distances, it does not use your labeling to
    train.

    In other words it returns what it thinks the label should be, not a
    prediction. Our accuracy in this example is therefore based on what the
    label acutally was versus what kmeans thinks the label should be.
    It's not really accuracy, though it could be used that way.

    kmeans also randomly assigns the label categories it uses, so it
    will be difficult to determine how they match the labels of the
    classes in y_ary.

    clf.labels_ is a dataframe of the labels kmeans has assigned.

    kmeans may be best when you have no idea of how to classify the data.
    
    """

    # Print the correct percentage
    print(correct/len(x_ary))


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
