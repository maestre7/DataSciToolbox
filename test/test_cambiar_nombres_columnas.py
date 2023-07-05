

from src.preprocessing import cambiar_nombres_columnas

import pandas as pd
import pytest



@pytest.fixture
def sample_dataframe():
    data = {'columna1': [1, 2, 3], 'columna2': [4, 5, 6]}
    return pd.DataFrame(data)

def test_cambiar_nombres_columnas(sample_dataframe):
    # Cambiar los nombres de las columnas
    df = cambiar_nombres_columnas(sample_dataframe, columna1='nueva_columna1', columna2='nueva_columna2')

    # Verificar que los nombres de las columnas se hayan cambiado correctamente
    assert 'nueva_columna1' in df.columns
    assert 'nueva_columna2' in df.columns
    assert 'columna1' not in df.columns
    assert 'columna2' not in df.columns

    # Verificar que los datos en las columnas se mantengan
    assert df['nueva_columna1'].equals(sample_dataframe['columna1'])
    assert df['nueva_columna2'].equals(sample_dataframe['columna2'])

