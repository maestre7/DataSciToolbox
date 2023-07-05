import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta
import os
import sys

# Agregar el directorio raíz del proyecto a la ruta de búsqueda de módulos
root_dir = os.path.dirname(os.path.dirname(os.path.abspath('preprocessing.py')))
sys.path.append(root_dir)

from src.preprocessing import obtener_hora_minuto_segundo

@pytest.fixture
def ejemplo_dataframe():
    return pd.DataFrame({"hora": ["08:30:45", "12:15:30"]})

def test_obtener_hora_minuto_segundo(ejemplo_dataframe):
    columna_hora = "hora"

    # Llamar a la función para obtener el nuevo DataFrame
    nuevo_df = obtener_hora_minuto_segundo(ejemplo_dataframe, columna_hora)

    # Verificar que las columnas "hora", "minuto" y "segundo" estén presentes en el nuevo DataFrame
    assert "hora" in nuevo_df.columns
    assert "minuto" in nuevo_df.columns
    assert "segundo" in nuevo_df.columns

    # Verificar que los valores en las columnas "hora", "minuto" y "segundo" sean correctos
    assert nuevo_df["hora"].tolist() == [8, 12]
    assert nuevo_df["minuto"].tolist() == [30, 15]
    assert nuevo_df["segundo"].tolist() == [45, 30]

def test_obtener_hora_minuto_segundo_columna_inexistente(ejemplo_dataframe):
    columna_hora_inexistente = "tiempo"

    # Verificar que se lance un ValueError si la columna de hora no existe en el DataFrame
    with pytest.raises(ValueError):
        obtener_hora_minuto_segundo(ejemplo_dataframe, columna_hora_inexistente)
