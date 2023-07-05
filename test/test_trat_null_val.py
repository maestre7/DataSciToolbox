import pandas as pd
import pytest


from src.preprocessing import tratar_valores_nulos

def test_tratar_valores_nulos():
    # Crear un DataFrame de prueba con valores nulos
    datos = {'A': [1, 2, None, 4, 5],
             'B': [None, 2, 3, 4, None],
             'C': [1, None, 3, None, 5]}
    dataframe = pd.DataFrame(datos)

    # Caso de prueba: eliminar valores nulos
    opcion_eliminar = 'eliminar'
    dataframe_sin_nulos = tratar_valores_nulos(dataframe, opcion_eliminar)
    assert dataframe_sin_nulos.isnull().sum().sum() == 0

    # Caso de prueba: rellenar valores nulos con cero
    opcion_rellenar_cero = 'rellenar_cero'
    dataframe_rellenado_cero = tratar_valores_nulos(dataframe, opcion_rellenar_cero)
    assert dataframe_rellenado_cero.isnull().sum().sum() == 0

    # Caso de prueba: rellenar valores nulos con la media
    opcion_rellenar_media = 'rellenar_media'
    dataframe_rellenado_media = tratar_valores_nulos(dataframe, opcion_rellenar_media)
    assert dataframe_rellenado_media.isnull().sum().sum() == 0

    # Caso de prueba: opción inválida
    opcion_invalida = 'opcion_invalida'
    dataframe_original = tratar_valores_nulos(dataframe, opcion_invalida)
    assert dataframe_original.equals(dataframe)