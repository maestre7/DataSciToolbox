import pandas as pd
import numpy as np
import seaborn as sns

def comprobacion_outliers(dataframe, nombre_columna):
    '''
    Esta función calcula el número de outliers y su proporción con respecto al total en una columna numérica de un DataFrame de Pandas.
    También muestra un gráfico boxplot utilizando la librería Seaborn para visualizar los outliers.

    Args:
    - dataframe: DataFrame de Pandas que contiene los datos.
    - nombre_columna: Nombre de la columna para la cual se desea detectar los outliers. Se deberá indicar en formato string.

    Return:
    - Gráfico boxplot generado por Seaborn.
    - Número de outliers en la columna especificada.
    - Porcentaje de outliers en relación al total de datos.
    '''
    try:
        if not isinstance(nombre_columna, str):
            raise TypeError("El nombre de la columna debe ser un string.")
            
        df = dataframe[nombre_columna]
        q1 = np.percentile(df, 25)
        q3 = np.percentile(df, 75)
        rango_intercuartilico = q3 - q1 
        outliers = df[(df < (q1 - 1.5 * rango_intercuartilico)) | (df > (q3 + 1.5 * rango_intercuartilico))]

        print("El número de outliers es de:", len(outliers))
        print("El porcentaje de Outliers es de:", round((len(outliers) / len(df)) * 100, 2), "%")

        return sns.boxplot(data=dataframe, x=nombre_columna, orient="h")

    except KeyError:
        print("Error: La columna especificada no existe en el DataFrame.")
    except TypeError as e:
        print("Error:", str(e))
    except:
        print("Error: Se produjo un problema al procesar la función. Por favor, revisa la documentación de la función, verifica que los parámetros de entrada estén correctamente indicados y revisa los datos de tu DataFrame.")
