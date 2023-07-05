import pandas as pd
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from balance_target_column_random import balance_target_column_random

def test_balance_target_column_random():
    # Create a dummy DataFrame for testing
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6],
                       'B': [6, 7, 8, 9, 10, 11],
                       'target': [0, 0, 0, 1, 1, 1]})

    # Call the balance_target_column_random function
    balanced_df = balance_target_column_random(df, 'target')

    # Perform assertions to check if the shapes and target balance are correct
    assert balanced_df.shape == (6, 3)
    assert balanced_df['target'].value_counts()[0] == 3
    assert balanced_df['target'].value_counts()[1] == 3