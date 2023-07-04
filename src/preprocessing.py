import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta

def segmentar_y_guardar(df, num_segmentos, output_folder):
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
    
    import pandas as pd

def obtener_hora_minuto_segundo(df, columna_hora):
    """
    Calcula la hora, minuto y segundo en columnas separadas a partir de una columna de hora en un DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame que contiene la columna de hora.
        columna_hora (str): Nombre de la columna que contiene la hora.

    Returns:
        pandas.DataFrame: DataFrame con las columnas "hora", "minuto" y "segundo" agregadas. Si ocurre un error durante la conversión, se devuelve el DataFrame original sin modificaciones.

    Raises:
        ValueError: Si la columna de hora no se encuentra en el DataFrame.

    Example:
        df = pd.DataFrame({"hora": ["08:30:45", "12:15:30"]})
        nuevo_df = obtener_hora_minuto_segundo(df, "hora")
    """
    if columna_hora not in df.columns:
        raise ValueError(f"La columna '{columna_hora}' no se encuentra en el DataFrame.")

    try:
        hora_dt = pd.to_datetime(df[columna_hora], format="%H:%M:%S")
        df["hora"] = hora_dt.dt.hour
        df["minuto"] = hora_dt.dt.minute
        df["segundo"] = hora_dt.dt.second
    except Exception as e:
        print("Error al convertir la columna de hora:", str(e))

    return df