import pandas as pd

def encoding_proporcional_target_binaria(dataframe: pd.DataFrame, target: str, columna_categorica: str, nueva_columna: str):
    '''
    Esta función realiza un encoding de una columna de tipo object en un DataFrame de pandas, creando una nueva columna. Esta función está diseñada para el contexto en el que la variable a predecir sea binaria.

    El encoding se realiza proporcionalmente al peso que cada variable categórica tiene en el problema.

    Argumentos:
    - dataframe: DataFrame de pandas que contiene los datos.
    - target: Nombre de la columna a predecir en el DataFrame. Debe ser binaria y se debe indicar como una cadena de texto.
    - columna_categorica: Nombre de la columna categórica que se desea encodear. Se debe indicar como una cadena de texto.
    - nueva_columna: Nombre de la nueva columna que contendrá los valores encodeados. Se debe indicar como una cadena de texto.
    '''

    try:
        dict_proporcional = dict(dataframe.groupby(columna_categorica)[target].mean())
        dataframe[nueva_columna] = dataframe[columna_categorica].map(dict_proporcional)
        return dataframe
    except (KeyError, TypeError) as e:
        print("Ocurrió un error al codificar la columna categórica:", str(e))
        return None