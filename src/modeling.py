1r6PoYic
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

from typing import Any
import numpy, pandas

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluacion_clas(nom_modelo: str, modelo: Any, X_train: numpy.ndarray, y_train: numpy.ndarray, X_test: numpy.ndarray, y_test: numpy.ndarray, redondeo: int = None) -> pandas.DataFrame: # type: ignore
    """
    Función para evaluar las predicciones de un modelo de clasificación de machine learning, devolviendo diferentes métricas en un dataframe.
    En caso de tratarse de una clasificación multiclase, el average de las métricas será None.
    Args:
        nom_modelo (str): El nombre del modelo.
        modelo (Any): Modelo de machine learning para hacer las predicciones.
        X_train (numpy.ndarray): Variables predictivas de entrenamiento.
        y_train (numpy.ndarray): Target de entrenamiento.
        X_test (numpy.ndarray): Variables predictivas de evaluación.
        y_test (numpy.ndarray): Target de evaluación.
        redondeo (int): Cantidad de decimales para redondear las métricas. (default: None)

    Returns:
        pandas.DataFrame: Dataframe con todas las métricas de evaluación del modelo.
    """
    try:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_pred_prob = modelo.predict_proba(X_test)
        
        if len(numpy.unique(y_test)) > 2:
            average = None
            multi_class = 'ovr'
        else:
            average = 'binary'
            multi_class = 'raise'
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)
        roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class=multi_class)
        
        if redondeo is not None:
            accuracy = numpy.round(accuracy, redondeo)
            precision = numpy.round(precision, redondeo)
            recall = numpy.round(recall, redondeo)
            f1 = numpy.round(f1, redondeo)
            roc_auc = numpy.round(roc_auc, redondeo)
        
        result_df = pandas.DataFrame(data=[[nom_modelo, accuracy, precision, recall, f1, roc_auc]], 
                                columns=["Model", 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
        return result_df
    
    except Exception as e:
        print("Error al evaluar el modelo'{}':".format(nom_modelo))
        return None # type: ignore
dev
