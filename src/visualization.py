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
    
def dist_variables(data: pd.DataFrame, target: str = None, ncols: int = 2, figsize: tuple = (30, 30)) -> plt.Figure: # type: ignore
    """
    Función para visualizar las distribuciones de las variables en un DataFrame.

    Args:
        data (pd.DataFrame): DataFrame con los datos.
        target (str or None): Nombre de la columna objetivo. Si es None, no se divide por grupos (default: None).
        ncols (int): Número de columnas en la figura de subplots (default: 2).
        figsize (tuple): Tamaño de la figura (default: (15, 20)).

    Returns:
        plt.Figure: Figura con todas las distribuciones de los datos.
    """
    try:
        total_columns = len(data.columns)
        nrows = int(np.ceil(total_columns / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.ravel() # type: ignore

        for i, column in enumerate(data.columns):
            ax = axes[i]

            if target is not None:
                # Variables categóricas
                if data[column].dtype == 'object':
                    sns.countplot(x=column, hue=target, data=data, ax=ax)
                    ax.set_title(column)
                    ax.legend(title=target)
                # Variables continuas
                else:
                    for value in data[target].unique():
                        sns.histplot(data[data[target] == value][column], ax=ax, label=value)
                    ax.set_title(column)
                    ax.legend(title=target)
            else:
                # Variables categóricas
                if data[column].dtype == 'object':
                    sns.countplot(x=column, data=data, ax=ax)
                    ax.set_title(column)
                # Variables continuas
                else:
                    sns.histplot(data[column], ax=ax)
                    ax.set_title(column)

        return fig

    except Exception as e:
        error_message = "Ocurrió un error durante la visualización: " + str(e)
        print(error_message)
        return None