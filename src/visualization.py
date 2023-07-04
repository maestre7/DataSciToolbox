import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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