import pandas as pd
import pytest

# Importa la función que deseas probar
from src.preprocessing import comprobacion_outliers

# Fixture para generar un DataFrame de prueba
@pytest.fixture
def dataframe_prueba():
    data = {
        'columna1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'columna2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    }
    return pd.DataFrame(data)

# Prueba para verificar el cálculo de outliers
def test_comprobacion_outliers(dataframe_prueba):
    # Llama a la función con la columna 'columna1'
    resultado = comprobacion_outliers(dataframe_prueba, 'columna1')
    
    # Verifica que el resultado sea un diccionario con las claves correctas
    assert isinstance(resultado, dict)
    assert 'numero_outliers' in resultado
    assert 'porcentaje_outliers' in resultado
    
    # Verifica que el número de outliers y el porcentaje sean correctos
    assert resultado['numero_outliers'] == 0
    assert resultado['porcentaje_outliers'] == 0.0

    # Llama a la función con la columna 'columna2'
    resultado = comprobacion_outliers(dataframe_prueba, 'columna2')

    # Verifica que el número de outliers y el porcentaje sean correctos
    assert resultado['numero_outliers'] == 0
    assert resultado['porcentaje_outliers'] == 0.0

# Prueba para verificar el manejo de errores
def test_comprobacion_outliers_errores(dataframe_prueba):
    # Llama a la función con una columna inexistente
    with pytest.raises(KeyError):
        comprobacion_outliers(dataframe_prueba, 'columna3')

    # Llama a la función con un nombre de columna no válido (no string)
    with pytest.raises(TypeError):
        comprobacion_outliers(dataframe_prueba, 123)

