import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.visualization import plot_pca_importance_agg

@pytest.fixture
def sample_data():
    # Crear datos de muestra para las pruebas
    data = pd.DataFrame({'Feat1': [1, 2, 3], 'Feat2': [4, 5, 6]})
    return data

@pytest.fixture
def sample_modelo_pca():
    # Crear un objeto de modelo de PCA de muestra para las pruebas
    modelo_pca = MockModeloPCA()
    modelo_pca.explained_variance_ratio_ = np.array([0.4, 0.6])
    return modelo_pca

class MockModeloPCA:
    def __init__(self):
        self.n_components_ = 2
        self.explained_variance_ratio_ = None

def test_plot_pca_importance_agg(sample_data, sample_modelo_pca, monkeypatch):
    # Monkeypatch la función `show` de plt para evitar la visualización del gráfico durante las pruebas
    monkeypatch.setattr(plt, 'show', lambda: None)

    # Llamar a la función con los datos de muestra y el modelo de PCA
    fig = plot_pca_importance_agg(sample_data, sample_modelo_pca)

    # Verificar si se devuelve el objeto Figure correctamente
    assert isinstance(fig, plt.Figure)

    # Realizar más aserciones para verificar el resultado esperado
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    assert len(ax.texts) == 2
    assert ax.get_ylim() == (0, 1.1)
    assert np.array_equal(ax.get_xticks(), np.array([1, 2]))
    assert ax.get_title() == 'Porcentaje de varianza explicada acumulada'
    assert ax.get_xlabel() == 'Componente principal'
    assert ax.get_ylabel() == '% varianza acumulada'