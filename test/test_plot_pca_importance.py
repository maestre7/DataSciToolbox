import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytest
from src.visualization import plot_pca_importance


@pytest.fixture
def sample_data():
    # Crear datos de muestra para las pruebas
    data = pd.DataFrame({'Feature1': [1, 2, 3],
                         'Feature2': [4, 5, 6],
                         'Feature3': [7, 8, 9]})
    return data

def test_plot_pca_importance(sample_data):
    # Crear un modelo de PCA ficticio para las pruebas
    n_components = 3
    explained_variance_ratio = [0.6, 0.3, 0.1]
    modelo_pca = type('MockPCA', (), {'n_components_': n_components, 'explained_variance_ratio_': explained_variance_ratio})()

    # Llamar a la función con los datos de muestra y el modelo de PCA ficticio
    fig = plot_pca_importance(sample_data, modelo_pca)

    # Verificar si se devuelve un objeto Figure
    assert isinstance(fig, plt.Figure)

    # Verificar si el título del gráfico es correcto
    assert fig.axes[0].get_title() == 'Porcentaje de varianza explicada por cada componente'

    # Verificar si el número de barras en el gráfico es igual al número de componentes principales
    assert len(fig.axes[0].patches) == n_components

    # Verificar si las etiquetas en las barras corresponden al porcentaje de varianza explicada
    for patch, ratio in zip(fig.axes[0].patches, explained_variance_ratio):
        label = patch.get_height()
        assert label == ratio

    # Verificar si los ejes están configurados correctamente
    assert fig.axes[0].get_xlabel() == 'Componente principal'
    assert fig.axes[0].get_ylabel() == 'Por. varianza explicada'