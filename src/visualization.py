import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_quality_counts(dataframe: pd.DataFrame, column:str, color:str) -> None:
    """Crea un gráfico de barras que muestra el recuento de valores en una columna específica.
    
    Args:
        dataframe (pd.DataFrame): El DataFrame que contiene los datos.
        column (str): El nombre de la columna para la cual se desea hacer el gráfico de barras.
        color (str): El color de las barras del gráfico.
    """
    # Crea el gráfico de barras utilizando value_counts() y plot.bar()
    try:
        dataframe[column].value_counts().plot.bar(rot=0, color=color)
        
        # Personaliza el gráfico
        plt.ylabel('Recuento')
        plt.xlabel(column)
        
        # Muestra el gráfico
        plt.show()

    except Exception as e:
        error_message = "Ocurrió un error durante la visualización: " + str(e)
        print(error_message)
        

def plot_heatmap(dataframe: pd.DataFrame, figsize:tuple) -> None:
    """Crea un mapa de calor (heatmap) a partir de un DataFrame.
    
    Args:
        dataframe (pd.DataFrame): El DataFrame que contiene los datos.
        figsize (tuple): El tamaño de la figura del heatmap en pulgadas (ancho, alto).
    """
    try: 

    # Configurar el tamaño de la figura
        plt.figure(figsize=figsize)
    
    # Crear el mapa de calor utilizando sns.heatmap()
        sns.heatmap(dataframe.corr(), annot=True)
    
    # Mostrar el mapa de calor
        plt.show()
    
    except Exception as e:
        error_message = "Ocurrió un error durante la visualización: " + str(e)
        print(error_message)
        


def plot_data(dataframe: pd.DataFrame, x: str, y: str, plot_type: str) -> None:
    """Visualiza diferentes tipos de gráficos a partir de un DataFrame.
    
    Args:
        dataframe (pd.DataFrame): El DataFrame que contiene los datos.
        x (str): La columna del DataFrame para el eje x.
        y (str): La columna del DataFrame para el eje y.
        plot_type (str): El tipo de gráfico a crear ('violin', 'scatter', 'bar', etc.).
    """
    try:

        if plot_type == 'violin':
            # Violin plot
            sns.violinplot(x=x, y=y, data=dataframe)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Violin Plot')
        
        elif plot_type == 'scatter':
            # Scatter plot
            plt.scatter(dataframe[x], dataframe[y])
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Scatter Plot')
        
        elif plot_type == 'bar':
            # Bar plot
            sns.barplot(x=x, y=y, data=dataframe)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Bar Plot')
        
        # Agregar más tipos de gráficos según sea necesario
        
        plt.show()

    except Exception as e:
        error_message = "Ocurrió un error durante la visualización: " + str(e)
        print(error_message)
        
