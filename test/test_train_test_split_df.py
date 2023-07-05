import pandas as pd
from sklearn.model_selection import train_test_split
from train_test_split_df import train_test_split_df

def test_train_test_split_df():
    # Create a dummy DataFrame for testing
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                       'B': [6, 7, 8, 9, 10],
                       'target': [0, 1, 0, 1, 1]})
    
    # Call the train_test_split_df function
    X_train, X_test, y_train, y_test = train_test_split_df(df, 'target', 0.2, 42)
    
    # Perform assertions to check if the shapes are correct
    assert X_train.shape[0] == 4
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 4
    assert y_test.shape[0] == 1
