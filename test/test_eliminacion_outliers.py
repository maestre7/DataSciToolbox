import pandas as pd
import numpy as np
import pytest

# Importar la función que se va a probar
from src.preprocessing import eliminacion_outliers


@pytest.fixture
def sample_dataframe():
    data = {
        'columna1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'columna2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    }
    return pd.DataFrame(data)

def test_eliminacion_outliers_valid_column(sample_dataframe):
    # Prueba con una columna válida
    columna = 'columna1'
    df = eliminacion_outliers(sample_dataframe, columna)
    assert len(df) == 10  # El DataFrame no debe cambiar ya que no hay outliers en columna1

def test_eliminacion_outliers_invalid_column(sample_dataframe):
    # Prueba con una columna no existente en el DataFrame
    columna = 'columna3'
    with pytest.raises(KeyError):
        eliminacion_outliers(sample_dataframe, columna)

def test_eliminacion_outliers_invalid_column_type(sample_dataframe):
    # Prueba con un tipo incorrecto para el nombre de la columna
    columna = 123
    with pytest.raises(TypeError):
        eliminacion_outliers(sample_dataframe, columna)
