import pandas as pd
import pytest
import os
import sys

# Agregar el directorio raíz del proyecto a la ruta de búsqueda de módulos
root_dir = os.path.dirname(os.path.dirname(os.path.abspath('preprocessing.py')))
sys.path.append(root_dir)

from src.preprocessing import eliminar_unidades_metricas

@pytest.fixture
def sample_dataframe():
    df = pd.DataFrame({
        'columna1': ['10 km', '20 kg', '30 hgz', '40 cv'],
        'columna2': ['50 km', '60 kg', '70 hgz', '80 cv']
    })
    return df

def test_eliminar_unidades_metricas(sample_dataframe):
    df = eliminar_unidades_metricas(sample_dataframe, 'columna1')

    assert df is not None
    assert 'columna1' in df.columns
    assert df['columna1'].dtype == float

    expected_values = [10.0, 20.0, 30.0, 40.0]
    assert df['columna1'].tolist() == expected_values

    # Verificar que la columna2 no haya sido modificada
    assert 'columna2' in df.columns
    assert df['columna2'].dtype != float
    assert df['columna2'].tolist() == sample_dataframe['columna2'].tolist()