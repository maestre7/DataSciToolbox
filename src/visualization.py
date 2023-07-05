import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from typing import Any, List
import folium
from wordcloud import WordCloud
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve
from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.metrics import roc_curve


def plot_moving_averages (data:pd.DataFrame, feature:str, medias_moviles=None, colores=None):
    '''
    Genera un gráfico interactivo utilizando la biblioteca Plotly que muestra las medias móviles de una característica específica en un conjunto de datos.

    Parámetros:
    - data (pd.DataFrame): El conjunto de datos que contiene la información.
    - feature (str): El nombre de la característica para la cual se calcularán las medias móviles y se mostrarán en el gráfico.
    - medias_moviles (list, opcional): Una lista de enteros que representan las ventanas de las medias móviles. Por defecto, se utilizan [8, 21, 30, 50, 100, 200].
    - colores (list, opcional): Una lista de colores en formato de cadena para asignar a las medias móviles. Debe tener la misma longitud que la lista de medias móviles. Por defecto, se utilizan ['orange', 'blue', 'grey', 'green', 'purple', 'red'].

    Retorna:
    - data (pd.DataFrame): El conjunto de datos original con las columnas de medias móviles agregadas.
    - fig (plotly.graph_objects.Figure): El gráfico interactivo generado con Plotly.

    Ejemplo de uso:
    data = pd.read_csv('datos.csv')
    plot_moving_averages(data, 'Precio', medias_moviles=[10, 20, 30], colores=['red', 'blue', 'green'])
    '''
    try:
        # Definir las medias móviles
        if medias_moviles is None:
            medias_moviles = [8, 21, 30, 50, 100, 200]
            
        # Definir los colores para las medias móviles
        if colores is None:
            colores = ['orange', 'blue', 'grey', 'green', 'purple', 'red']

        # Calcular las medias móviles
        for ma in medias_moviles:
            columna = 'MA_' + str(ma)  # Nombre de la columna de la media móvil
            data[columna] = data[feature].rolling(window=ma).mean()

        # Crear la figura de Plotly
        fig = go.Figure()

        # Agregar 'feature' al gráfico
        fig.add_trace(go.Scatter(x=data.index, y=data[feature], name='Precio', line=dict(color='blue', width=1)))

        # Agregar las medias móviles al gráfico con colores diferentes
        for ma, color in zip(medias_moviles, colores):
            columna = 'MA_' + str(ma)
            fig.add_trace(go.Scatter(x=data.index, y=data[columna], name='MA ' + str(ma), line=dict(color=color, width=0.5)))

        # Personalizar el diseño del gráfico
        fig.update_layout(
            title={'text': 'Medias Móviles', 'x': 0.5, 'xanchor': 'center'},
            xaxis_title='Fecha',
            yaxis_title=feature,
        )

        # Mostrar el gráfico interactivo
        fig.show()

        return data, fig
    
    except Exception as e:
        print("Error en la función plot_moving_averages:", str(e))


def plot_pca_importance (data, modelo_pca):
    '''
    Genera un gráfico de barras que muestra el porcentaje de varianza explicada por cada componente principal
    en un modelo de PCA.

    Parámetros:
    - data: El conjunto de datos utilizado en el modelo de PCA.
    - modelo_pca: El objeto del modelo de PCA entrenado.

    Retorna:
    - fig: El objeto Figure del gráfico generado.

    '''
    try:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.bar(
            x      = np.arange(modelo_pca.n_components_) + 1,
            height = modelo_pca.explained_variance_ratio_
        )

        for x, y in zip(np.arange(len(data.columns)) + 1, modelo_pca.explained_variance_ratio_):
            label = round(y, 2)
            ax.annotate(
                label,
                (x,y),
                textcoords="offset points",
                xytext=(0,10),
                ha='center'
            )

        ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
        ax.set_ylim(0, 1.1)
        ax.set_title('Porcentaje de varianza explicada por cada componente')
        ax.set_xlabel('Componente principal')
        ax.set_ylabel('Por. varianza explicada')

        fig.show()

        return fig
    
    except Exception as e:
        print("Ocurrió un error en plot_pca_importance:", str(e))


def plot_pca_importance_agg (data, modelo_pca):
    '''
    Genera un gráfico de línea que muestra el porcentaje acumulado de varianza explicada por cada componente principal
    en un modelo de PCA.

    Parámetros:
    - data: El conjunto de datos utilizado en el modelo de PCA.
    - modelo_pca: El objeto del modelo de PCA entrenado.

    Retorna:
    - fig: El objeto Figure del gráfico generado.

    '''
    try:
        prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(
            np.arange(len(data.columns)) + 1,
            prop_varianza_acum,
            marker = 'o'
        )

        for x, y in zip(np.arange(len(data.columns)) + 1, prop_varianza_acum):
            label = round(y, 2)
            ax.annotate(
                label,
                (x,y),
                textcoords="offset points",
                xytext=(0,10),
                ha='center'
            )
            
        ax.set_ylim(0, 1.1)
        ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
        ax.set_title('Porcentaje de varianza explicada acumulada')
        ax.set_xlabel('Componente principal')
        ax.set_ylabel('% varianza acumulada')

        fig.show()

        return fig
    

    except Exception as e:
        print("Ocurrió un error en plot_pca_importance_agg:", str(e))
        

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
    
def plot_correlations(dataframe, target, figsize, filename):
    
    try:
        """Esta función crea un gráfico de barras que muestra las correlaciones de la variable objetivo con el resto de las variables.

        Args:
        dataframe (pd.DataFrame): El DataFrame que contiene los datos.
        target (str): El nombre de la columna que representa la variable objetivo.
        figsize (tuple): El tamaño de la figura del gráfico en pulgadas (ancho, alto).
        filename (str): El nombre del archivo en el que se guardará la figura.
        """

        #para calcular las correlaciones de la variable objetivo con el resto de las variables con dataframe.corr()
        correlations = dataframe.corr()[target].drop(target)

        #Ordenar las correlaciones de mayor a menor usando correlations.sort_values()
        correlations = correlations.sort_values(ascending=False)

        #tamaño de la figura
        plt.figure(figsize=figsize)

        #gráfico de barras con sns.barplot()
        sns.barplot(x=correlations, y=correlations.index, palette='Blues')

        #Añadir los ejes y el título usando plt.xlabel(), plt.ylabel() y plt.title()
        plt.xlabel('Correlación')
        plt.ylabel('Variable')
        plt.title(f'Correlaciones de {target} con el resto de las variables')

        #Guardar la figura en un archivo
        plt.savefig(filename)

        #gráfico
        plt.show()
    
    except Exception as e:
        error_message = "Ocurrió un error durante la visualización: " + str(e)
        print(error_message)
        return None

def plot_map(dataframe, figsize):
    
    try:
        """Esta función crea un mapa interactivo a partir de un DataFrame.

        Argumentos:
            dataframe (pd.DataFrame): El DataFrame que contiene los datos.
            figsize (tuple): El tamaño de la figura del mapa en píxeles (ancho, alto).
        """

        #Crear el mapa usando folium.Map()
        mapa = folium.Map(location=[dataframe['lat'].mean(), dataframe['lon'].mean()], 
                        zoom_start=10, tiles='Stamen Terrain')

        #Añadir marcadores con folium.Marker()
        for i, row in dataframe.iterrows():
            folium.Marker(location=[row['lat'], row['lon']],
                        popup=row['name'], 
                        icon=folium.Icon(color=row['color'])).add_to(mapa)

        #Mostrar el mapa usando folium.Figure()
        fig = folium.Figure(width=figsize[0], height=figsize[1])
        fig.add_child(mapa)
        mapa.save("map.html") #PARA QUE SE VEA EL MAPA
        fig
        
    except Exception as e:
        error_message = "Ocurrió un error durante la visualización: " + str(e)
        print(error_message)
        return None
    
def plot_wordcloud(text, figsize, filename):
    try:
        """Esta función crea una nube de palabras a partir de un texto.

        Argumentos=
            text (str): El texto que contiene las palabras.
            figsize (tuple): El tamaño de la figura de la nube de palabras en pulgadas (ancho, alto).
            filename (str): El nombre del archivo en el que se guardará la figura.
        """

        #Nube de palabras creada usando WordCloud()
        wc = WordCloud(background_color='white', 
                    max_words=100, 
                    width=figsize[0]*100, 
                    height=figsize[1]*100).generate(text)

        #Tamaño de la figura
        plt.figure(figsize=figsize)

        #Mostrar la nube de palabras usando plt.imshow() y plt.axis()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')

        #Guardar
        plt.savefig(filename)

        #Mostrar
        plt.show()
    except Exception as e:
        error_message = "Ocurrió un error durante la visualización: " + str(e)
        print(error_message)
        return None
