import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
    ax.set_ylabel('Por. varianza explicada')