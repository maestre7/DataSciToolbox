import pandas as pd
import numpy as np

def eliminacion_outliers(dataframe:pd.DataFrame, nombre_columna:str):
    '''
    Esta función elimina las filas del DataFrame que contienen valores atípicos (outliers) en una columna especificada.

    Args:
    - dataframe: DataFrame de Pandas que contiene los datos.
    - nombre_columna: Nombre de la columna en la cual se desean eliminar las filas con outliers. Se deberá indicar en formato string.

    Return:
    - Devuelve el DataFrame sin los valores atípicos de la columna especificada.
    '''

    try:
        if not isinstance(nombre_columna, str):
            raise TypeError("El nombre de la columna debe ser un string.")
        
        df = dataframe.copy()
        q1 = np.percentile(df[nombre_columna], 25)
        q3 = np.percentile(df[nombre_columna], 75)
        rango_intercuartilico = q3 - q1 
        df = df[(df[nombre_columna] >= (q1 - 1.5 * rango_intercuartilico)) & (df[nombre_columna] <= (q3 + 1.5 * rango_intercuartilico))]

        return df

    except KeyError:
        print("Error: La columna especificada no existe en el DataFrame.")
    except TypeError as e:
        print("Error:", str(e))
    except:
        print("Error: Se produjo un problema al procesar la función. Por favor, revisa la documentación de la función, verifica que los parámetros de entrada estén correctamente indicados y revisa los datos de tu DataFrame.")
