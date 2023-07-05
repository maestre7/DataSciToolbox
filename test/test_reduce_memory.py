import pandas as pd
import pytest
from src.preprocessing import ReduceMemory

@pytest.fixture
def sample_data():
    data = {
        'col1': ['A', 'B', 'C'],
        'col2': [1.5, 2.3, 3.1],
        'col3': [100, 200, 300]
    }
    return pd.DataFrame(data)

def test_reduce_memory(sample_data):
    reducer = ReduceMemory()
    reduced_data = reducer.process(sample_data)

    # Verificar que el tamaño después de la reducción es menor que el tamaño antes de la reducción
    for col in reduced_data.columns:
        before_size = reducer.before_size
        after_size = reducer.after_size
        assert after_size < before_size

    # Verificar que los tipos de datos se hayan reducido correctamente
    assert reduced_data['col1'].dtype == 'int8'
    assert reduced_data['col2'].dtype == 'float16'
    assert reduced_data['col3'].dtype == 'int16'
