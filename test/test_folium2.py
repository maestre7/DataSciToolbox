import pytest
from src.visualization import plot_map
import pandas as pd

def test_plot_map():
    # Caso de prueba con datos válidos
    dataframe = pd.DataFrame({'lat': [43.3183, 43.3183], 
                              'lon': [-1.9812, -1.9812], 
                              'name': ['San Sebastián', 'Donostia'], 
                              'color': ['red', 'blue']})
    figsize = (800, 600)
    plot_map(dataframe, figsize)

    # Caso de prueba con datos inválidos
    with pytest.raises(Exception):
        dataframe = pd.DataFrame({'lat': [43.3183, 43.3183], 
                                  'lon': [-1.9812, -1.9812], 
                                  'name': ['San Sebastián', 'Donostia'], 
                                  'color': ['red', 'blue']})
        figsize = (800, 600)
        plot_map(dataframe, figsize)

    #assert True == True

