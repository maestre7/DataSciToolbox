import pandas as pd
import numpy as np
import pytest

# Importar la función que se va a probar
from src.preprocessing import eliminacion_outliers


# Crear pruebas utilizando pytest
def test_eliminacion_outliers():
    # Crear un DataFrame de ejemplo
    data = {
        'columna1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'columna2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    }
    df = pd.DataFrame(data)

    # Ejecutar la función y obtener el DataFrame resultante
    df_resultado = eliminacion_outliers(df, 'columna1')

    # Comprobar que el DataFrame resultante no tiene outliers
    assert len(df_resultado) == 15

    # Comprobar que el DataFrame resultante tiene solo los valores esperados
    assert df_resultado['columna1'].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # Comprobar que la columna especificada no existe en el DataFrame
    with pytest.raises(KeyError):
        eliminacion_outliers(df, 'columna3')

    # Comprobar que el nombre de la columna es un string
    with pytest.raises(TypeError):
        eliminacion_outliers(df, 123)

    # Comprobar que se produce un error con un DataFrame vacío
    with pytest.raises(Exception):
        eliminacion_outliers(pd.DataFrame(), 'columna1')

