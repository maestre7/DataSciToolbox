import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta
import os
import sys

# Agregar el directorio raíz del proyecto a la ruta de búsqueda de módulos
root_dir = os.path.dirname(os.path.dirname(os.path.abspath('preprocessing.py')))
sys.path.append(root_dir)

from src.preprocessing import calcular_edad

@pytest.fixture
def sample_data():
    # Crear un DataFrame de ejemplo con datos de prueba
    data = {
        "name": ["John", "Jane", "Michael"],
        "dob": ["1990-01-01", "1985-03-15", "1998-07-20"],
    }
    df = pd.DataFrame(data)
    return df

def test_calcular_edad(sample_data):
    # Obtener el DataFrame de muestra
    df = sample_data.copy()

    # Calcular la edad utilizando la función calcular_edad
    df_con_edad = calcular_edad(df, "dob")

    # Verificar que la columna "edad" se haya agregado correctamente
    assert "edad" in df_con_edad.columns

    # Verificar que la edad se haya calculado correctamente para cada registro
    fecha_referencia = pd.to_datetime("today").date()
    for index, row in df_con_edad.iterrows():
        dob = pd.to_datetime(row["dob"]).date()
        edad_esperada = relativedelta(fecha_referencia, dob).years
        assert row["edad"] == edad_esperada
