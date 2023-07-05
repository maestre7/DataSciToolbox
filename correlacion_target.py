import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlations(dataframe, target, figsize, filename):
    
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
