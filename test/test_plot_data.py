from funciones_github import plot_data
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
        'x': [1, 2, 3, 4, 5],
        'y': [5, 4, 3, 2, 1]
    }
    return pd.DataFrame(data)

def test_plot_data(sample_dataframe):
    x = 'x'
    y = 'y'
    plot_type = 'scatter'
    
    # Capturar la salida del gráfico
    plt.figure()
    with pytest.raises(SystemExit):
        plot_data(sample_dataframe, x, y, plot_type)
    plt.close()

    # Realizar las aserciones necesarias
    # Ejemplo: verificar que el tipo de gráfico se haya creado correctamente
    assert plt.gca().get_title() == 'Scatter Plot'
