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
        print(f"Ocurrió un error: {str(e)}")

def plot_heatmap(dataframe: pd.DataFrame, figsize: tuple) -> None:
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
        print(f"Ocurrió un error: {str(e)}")




def plot_learning_curve(estimator, X, y, cv=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Genera una curva de aprendizaje para un estimador dado.

    Parámetros:
    estimator: objeto de estimador
    X: matriz de características
    y: vector de etiquetas
    cv: validación cruzada (opcional)
    train_sizes: tamaños de los conjuntos de entrenamiento (opcional)

    Devuelve:
    None
    """
    try:
        plt.figure()
        plt.title("Curva de aprendizaje")
        plt.xlabel("Tamaño del conjunto de entrenamiento")
        plt.ylabel("Score")

        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes)

        # Calcular la media y el desvío estándar del training score
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)

        # Graficar la curva de aprendizaje del training score
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")

        plt.legend(loc="best")
    except Exception as e:
        print(f"An error occurred: {e}")


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
        

def plot_precision_recall_curve(y_true, y_prob):
    """
    Genera una curva de precisión-recall dadas las etiquetas verdaderas y las probabilidades predichas.

    Parámetros:
    y_true: vector de etiquetas verdaderas
    y_prob: vector de probabilidades predichas
    """
    try:
        # Verificar si los argumentos son válidos
        y_true = column_or_1d(y_true)
        y_prob = column_or_1d(y_prob)
        check_consistent_length(y_true, y_prob)

        # Calcular la precisión y el recall para diferentes umbrales
        precision, recall, _ = precision_recall_curve(y_true, y_prob)

        # Generar la gráfica
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


def plot_quality_counts(dataframe: pd.DataFrame, column: str, color: str) -> None:
    """Crea un gráfico de barras que muestra el recuento de valores en una columna específica.
    
    Args:
        dataframe (pd.DataFrame): El DataFrame que contiene los datos.
        column (str): El nombre de la columna para la cual se desea hacer el gráfico de barras.
        color (str): El color de las barras del gráfico.
    """
    try:
        # Crea el gráfico de barras utilizando value_counts() y plot.bar()
        dataframe[column].value_counts().plot.bar(rot=0, color=color)
        
        # Personaliza el gráfico
        plt.ylabel('Recuento')
        plt.xlabel(column)
        
        # Muestra el gráfico
        plt.show()
    except KeyError:
        print(f"La columna '{column}' no existe en el DataFrame.")
    except Exception as e:
        print(f"Ocurrió un error: {str(e)}")


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
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
    except Exception as e:
        print(f'Error: {e}')

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
        plt.scatter(y_test, predictions, color='blue',
                    label='Valores de prueba vs. Valores predichos')
        # Línea de referencia: valores reales = valores predichos
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
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










