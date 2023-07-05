
import pandas as pd
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

def balance_target_column_smote(df, target_column):
    """
    Equilibra una columna objetivo especificada en el DataFrame utilizando una combinación de sobremuestreo y submuestreo.
    Los datos se mezclan antes de devolver el DataFrame equilibrado.

    Parámetros:
        df (pandas.DataFrame): El DataFrame de entrada que contiene la columna objetivo.
        target_column (str): El nombre de la columna objetivo a equilibrar.

    Retorna:
        pandas.DataFrame: Un nuevo DataFrame con la columna objetivo equilibrada y mezclada.

    """
    try:
        # Separar características (X) y variable objetivo (y)
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Instanciar el muestreador SMOTEENN
        sampler = SMOTEENN(random_state=42)

        # Remuestrear los datos
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Crear un nuevo DataFrame equilibrado
        balanced_df = pd.concat([X_resampled, y_resampled], axis=1)

        # Mezclar los datos
        balanced_df = shuffle(balanced_df, random_state=42)

        return balanced_df

    except Exception as e:
        print("Error al equilibrar la columna objetivo:", str(e))
        return None