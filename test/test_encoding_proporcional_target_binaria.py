import pandas as pd
import pytest
from preprocessing import encoding_proporcional_target_binaria

@pytest.fixture
def sample_dataframe():
    data = {
        'columna_categorica': ['A', 'B', 'A', 'B', 'C', 'C'],
        'target': [1, 1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

def test_encoding_proporcional_target_binaria(sample_dataframe):
    encoded_dataframe = encoding_proporcional_target_binaria(sample_dataframe, 'target', 'columna_categorica', 'encoded_column')
    
    # Comprobar si se creó la nueva columna
    assert 'encoded_column' in encoded_dataframe.columns
    
    # Comprobar los valores encodeados
    expected_values = [0.5, 0.6666666666666666, 0.5, 0.6666666666666666, 0.3333333333333333, 0.3333333333333333]
    assert encoded_dataframe['encoded_column'].tolist() == expected_values
    
    # Comprobar que los valores encodeados son proporcionales a la variable target
    assert encoded_dataframe['encoded_column'][0] == pytest.approx(0.5)
    assert encoded_dataframe['encoded_column'][1] == pytest.approx(0.6666666666666666)
    assert encoded_dataframe['encoded_column'][2] == pytest.approx(0.5)
    assert encoded_dataframe['encoded_column'][3] == pytest.approx(0.6666666666666666)
    assert encoded_dataframe['encoded_column'][4] == pytest.approx(0.3333333333333333)
    assert encoded_dataframe['encoded_column'][5] == pytest.approx(0.3333333333333333)
    
    # Comprobar el manejo de errores
    invalid_dataframe = pd.DataFrame({'columna_categorica': [1, 2, 3], 'target': [0, 1, 0]})
    assert encoding_proporcional_target_binaria(invalid_dataframe, 'target', 'columna_categorica', 'encoded_column') is None
