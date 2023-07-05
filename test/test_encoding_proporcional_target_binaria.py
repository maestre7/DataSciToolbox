import pandas as pd
import pytest
from src.preprocessing import encoding_proporcional_target_binaria

# Crear una función de prueba utilizando pytest
def test_encoding_proporcional_target_binaria():
    # Crear un DataFrame de ejemplo
    data = {'Columna_Categorica': ['A', 'B', 'A', 'B', 'C'],
            'Target': [1, 0, 1, 1, 0]}
    df = pd.DataFrame(data)

    # Llamar a la función que se va a probar
    encoded_df = encoding_proporcional_target_binaria(df, 'Target', 'Columna_Categorica', 'Nueva_Columna')

    # Comprobar si la columna nueva fue creada correctamente
    assert 'Nueva_Columna' in encoded_df.columns

    # Comprobar si los valores de la columna nueva son correctos utilizando pytest.approx()
    expected_values = [1.0, 0.5, 1.0, 0.5, 0.0]
    assert encoded_df['Nueva_Columna'].tolist() == pytest.approx(expected_values)

    # Probar un caso donde la columna target no existe
    encoded_df = encoding_proporcional_target_binaria(df, 'Nonexistent', 'Columna_Categorica', 'Nueva_Columna')
    assert encoded_df is None

    # Probar un caso donde la columna columna_categorica no existe
    encoded_df = encoding_proporcional_target_binaria(df, 'Target', 'Nonexistent', 'Nueva_Columna')
    assert encoded_df is None

    # Probar un caso donde la columna target no es binaria
    data = {'Columna_Categorica': ['A', 'B', 'A', 'B', 'C'],
            'Target': [1, 0, 1, 2, 0]}
    df = pd.DataFrame(data)
    encoded_df = encoding_proporcional_target_binaria(df, 'Target', 'Columna_Categorica', 'Nueva_Columna')
    assert encoded_df is None
