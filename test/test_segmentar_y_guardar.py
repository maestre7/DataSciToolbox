import os
import pandas as pd
import pytest
import sys

# Agregar el directorio raíz del proyecto a la ruta de búsqueda de módulos
root_dir = os.path.dirname(os.path.dirname(os.path.abspath('preprocessing.py')))
sys.path.append(root_dir)

from src.preprocessing import segmentar_y_guardar

@pytest.fixture
def df_ejemplo():
    # DataFrame de ejemplo para las pruebas
    data = {'A': [1, 2, 3, 4, 5, 6], 'B': ['a', 'b', 'c', 'd', 'e', 'f']}
    return pd.DataFrame(data)

@pytest.fixture
def ruta_salida(tmpdir):
    # Ruta de la carpeta de salida para las pruebas
    return tmpdir.mkdir("output")

def test_segmentar_y_guardar(df_ejemplo, ruta_salida):
    # Llamar a la función con el DataFrame de ejemplo y 2 segmentos
    segmentar_y_guardar(df_ejemplo, 2, str(ruta_salida))

    # Verificar que se hayan creado 2 archivos CSV en la carpeta de salida
    assert len(os.listdir(ruta_salida)) == 2

    # Verificar el contenido de los archivos CSV
    for i in range(1, 3):
        archivo_csv = f'{ruta_salida}/segmento_{i}.csv'
        assert os.path.exists(archivo_csv)

        # Leer el archivo CSV y comparar con el segmento correspondiente del DgataFrame original
        df_segmento = pd.read_csv(archivo_csv)
        df_segmento_expected = df_ejemplo.iloc[(i - 1) * 3:i * 3].reset_index(drop=True)
        assert df_segmento.equals(df_segmento_expected)
