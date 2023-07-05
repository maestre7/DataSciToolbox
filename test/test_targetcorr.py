import pytest
from src.visualization import plot_correlations
import pandas as pd

def test_plot_correlations():
    # Caso de prueba con datos válidos
    dataframe = pd.DataFrame({'var1': [1, 2, 3], 
                              'var2': [4, 5, 6], 
                              'var3': [7, 8, 9]})
    target = 'var1'
    figsize = (8, 6)
    filename = 'test.png'
    plot_correlations(dataframe, target, figsize, filename)

    # Caso de prueba con datos inválidos
    with pytest.raises(Exception):
        dataframe = pd.DataFrame({'var1': [1, 2, 3], 
                                  'var2': [4, 5, 'invalid'], 
                                  'var3': [7, 8, 9]})
        target = 'var1'
        figsize = (8, 6)
        filename = 'test.png'
        plot_correlations(dataframe, target, figsize, filename)

