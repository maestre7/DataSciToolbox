import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Any, List

def plot_scatter_with_reference(y_test: np.array, predictions: np.array, title: str) -> None:
    """
    Genera un gráfico de dispersión con una línea de referencia para comparar los valores de prueba con los valores predichos.

    Args:
        y_test (array-like): Valores de prueba.
        predictions (array-like): Valores predichos.
        title (str): Título del gráfico.

    Returns:
        None
    """
    try:
        sns.set(style="darkgrid")
        plt.scatter(y_test, predictions, color='blue', label='Valores de prueba vs. Valores predichos')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Línea de referencia: valores reales = valores predichos
        plt.scatter(y_test, y_test, color='red', label='Valores de prueba')
        plt.xlabel('Valores de prueba')
        plt.ylabel('Valores predichos')
        plt.title(title)
        plt.legend()
        plt.show()
    except Exception as e:
        print("Ocurrió un error al generar el gráfico de dispersión:", str(e))
    finally:
        return None