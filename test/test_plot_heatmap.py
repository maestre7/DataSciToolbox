from funciones_github import plot_heatmap
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
        'column1': [1, 2, 3],
        'column2': [4, 5, 6],
        'column3': [7, 8, 9]
    }
    return pd.DataFrame(data)

def test_plot_heatmap(sample_dataframe):
    figsize = (8, 6)
    
    # Capturar la salida del gráfico
    plt.figure()
    with pytest.raises(SystemExit):
        plot_heatmap(sample_dataframe, figsize)
    plt.close()

    # Realizar las aserciones necesarias
    # Ejemplo: verificar que el mapa de calor se haya creado correctamente
    assert plt.gcf().get_size_inches() == figsize

