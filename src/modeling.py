
import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_df(df, target_col, test_percent, random_state):
    """
    Divide un DataFrame en conjuntos de entrenamiento y prueba para el aprendizaje automático.

    Parámetros:
        df (pandas.DataFrame): El DataFrame de entrada.
        target_col (str): El nombre de la columna objetivo en el DataFrame.
        test_percent (float): El porcentaje de datos para usar en la prueba (entre 0 y 1).
        random_state (int): El estado aleatorio para garantizar la reproducibilidad.

    Retorna:
        X_train (pandas.DataFrame): Las características del conjunto de entrenamiento.
        X_test (pandas.DataFrame): Las características del conjunto de prueba.
        y_train (pandas.Series): La variable objetivo del conjunto de entrenamiento.
        y_test (pandas.Series): La variable objetivo del conjunto de prueba.

    Imprime:
        La forma de cada conjunto para confirmar la dimensión.
    """
    try:
        # Separar características (X) y variable objetivo (y)
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=random_state)

        print("Forma de X:", X.shape)
        print("Forma de y:", y.shape)

        print("Forma de X_train:", X_train.shape)
        print("Forma de X_test:", X_test.shape)
        print("Forma de y_train:", y_train.shape)
        print("Forma de y_test:", y_test.shape)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print("Error al dividir los datos en conjuntos de entrenamiento y prueba:", str(e))
        return None, None, None, None