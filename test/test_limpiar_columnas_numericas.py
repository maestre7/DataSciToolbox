import pandas as pd
import pytest

from src.preprocessing import limpiar_columnas_numericas

# Crear un DataFrame de prueba
@pytest.fixture
def dataframe_prueba():
    data = {
        'Columna1': ['1', '2', '3', '4'],
        'Columna2': ['5', '6', '7', '8'],
        'Columna3': ['9', '10', '11', '12'],
    }
    return pd.DataFrame(data)

# Prueba para verificar el manejo de errores
def test_manjo_errores(dataframe_prueba):
    # Definir caracteres especiales inválidos que causarán un error
    caracteres_especiales = ['$', ',,']
    valor_reemplazo = ''
    
    try:
        limpiar_columnas_numericas(dataframe_prueba, 'Columna1', caracteres_especiales, valor_reemplazo)
    except Exception as e:
        # Aquí puedes realizar acciones específicas en caso de que se capture la excepción
        pytest.fail(f"Se esperaba una excepción, pero se capturó: {str(e)}")








