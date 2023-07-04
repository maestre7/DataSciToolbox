import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_df(df, target_col, test_percent, random_state):
    """
    Split a DataFrame into training and testing sets for machine learning.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        target_col (str): The name of the target column in the DataFrame.
        test_percent (float): The percentage of data to use for testing (between 0 and 1).
        random_state (int): The random state to ensure reproducibility.

    Returns:
        X_train (pandas.DataFrame): The training set features.
        X_test (pandas.DataFrame): The testing set features.
        y_train (pandas.Series): The training set target variable.
        y_test (pandas.Series): The testing set target variable.

    Prints:
        Shape of each set to confirm dimension.
    """
    # Separate features (X) and target variable (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=random_state)

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape) 

    return X_train, X_test, y_train, y_test
