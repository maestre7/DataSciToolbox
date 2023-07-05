import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta

def segmentar_y_guardar(df: pd.DataFrame, num_segmentos:int, output_folder:str):
    """
    Segmenta un DataFrame en varios segmentos y los guarda en archivos CSV.

    Args:
        df (pandas.DataFrame): DataFrame a segmentar.
        num_segmentos (int): Número de segmentos en los que dividir el DataFrame.
        output_folder (str): Ruta de la carpeta de salida para guardar los archivos CSV.

    Returns:
        None
    """
    try:
        segmentos = np.array_split(df, num_segmentos)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i, segmento in enumerate(segmentos):
            segmento.to_csv(f'{output_folder}/segmento_{i+1}.csv', index=False)

        print(f"Se han creado {num_segmentos} archivos CSV segmentados en la carpeta '{output_folder}'.")
    except Exception as e:
        print(f"Error al guardar los segmentos en archivos CSV: {str(e)}")

def calcular_edad(df: pd.DataFrame, columna_nacimiento: str, fecha_referencia: str = None) -> pd.DataFrame:
    """
    Calcula la edad a partir de la fecha de nacimiento en un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        columna_nacimiento (str): El nombre de la columna que contiene las fechas de nacimiento.
        fecha_referencia (str, optional): La fecha de referencia para calcular la edad. Si no se proporciona,
            se utilizará la fecha actual. Formato: 'YYYY-MM-DD'. Default: None.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con una columna adicional "edad" que contiene las edades calculadas.

    Raises:
        ValueError: Si la columna de fecha de nacimiento no existe en el DataFrame.

    Example:
        # Cargar el DataFrame desde algún origen de datos
        df = pd.read_csv("datos.csv")

        # Calcular la edad utilizando la función calcular_edad
        df_con_edad = calcular_edad(df, "dob")

        # Imprimir el DataFrame resultante con la columna "edad" agregada
        print(df_con_edad)
    """
    try:
        if columna_nacimiento not in df.columns:
            raise ValueError(f"La columna '{columna_nacimiento}' no existe en el DataFrame.")

        df = df.copy()
        df[columna_nacimiento] = pd.to_datetime(df[columna_nacimiento])

        if fecha_referencia is None:
            fecha_referencia = pd.to_datetime("today").date()
        else:
            fecha_referencia = pd.to_datetime(fecha_referencia).date()

        df["edad"] = df[columna_nacimiento].apply(lambda x: relativedelta(fecha_referencia, x).years)
        return df
    except Exception as e:
        print(f"Ocurrió un error: {str(e)}")
        return df