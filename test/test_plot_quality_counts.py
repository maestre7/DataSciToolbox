from funciones_github import plot_quality_counts
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pytest
import warnings

@pytest.fixture
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
# Definir una fixture de ejemplo con datos de prueba

def sample_dataframe():
    data = {
        'column1': ['A', 'B', 'A', 'C', 'B', 'B'],
        'column2': [1, 2, 3, 1, 2, 1]
    }
    return pd.DataFrame(data)

def test_plot_quality_counts(sample_dataframe):
    column = 'column1'
    color = 'blue'
    
    # Capturar la salida del gráfico
    plt.figure()
    with pytest.raises(SystemExit):
        plot_quality_counts(sample_dataframe, column, color)
    plt.close()

    # Realizar las aserciones necesarias
    # Ejemplo: verificar que el título del gráfico sea correcto
    assert plt.gca().get_title() == 'Gráfico de barras para la columna column1'