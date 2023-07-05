import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns

from src.visualization import plot_scatter_with_reference

def test_plot_scatter_with_reference():
    # Generar datos de prueba
    y_test = [1, 2, 3, 4, 5]
    predictions = [2, 4, 1, 3, 7]
    title = 'Gráfico de dispersión'

    # Ejecutar la función
    plot_scatter_with_reference(y_test, predictions, title)

    # Comprobar si se generó el gráfico sin errores
    assert plt.gcf().number == 1