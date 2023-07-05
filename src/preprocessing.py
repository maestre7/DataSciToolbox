import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta
import re
from typing import Union
import sys
import rarfile
import zipfile
import yfinance as yf

class ReduceMemory:
    """
    Clase para reducir el consumo de memoria de un DataFrame de Pandas
    """

    def __init__(self) -> None:
        self.before_size = 0
        self.after_size = 0
        self.sumary = {}

    def process(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce el consumo de memoria de un DataFrame de Pandas aplicando reducciones de 
        tipo de datos en cada columna del DataFrame.

        Args:
            data_df (pd.DataFrame): DataFrame de Pandas a procesar.

        Returns:
            pd.DataFrame: DataFrame de Pandas con el consumo de memoria reducido.
        """
        
        try:
            self.sumary = {}
            cols = data_df.columns

            for col in cols:
                dtype = data_df[col].dtype

                if dtype == 'object':
                    data_df[col] = self.reduce_object(data_df[col])
                    self.sumary_size(col)
                elif dtype == 'float':
                    data_df[col] = self.reduce_float(data_df[col])
                    self.sumary_size(col)
                elif dtype == 'int':
                    data_df[col] = self.reduce_int(data_df[col])
                    self.sumary_size(col)

        except Exception as err:
            print(f"Error processing column '{col}': {str(err)}")
            return False

        return data_df
    
    def sumary_size(self, col: str):
        """
        Registra el resumen del tamaño antes y después de la reducción de una columna.

        Args:
            col (str): Nombre de la columna.
        """

        try:
            self.sumary[col] = {"before_size": self.before_size, 
                                "after_size": self.after_size,}
        except Exception as err:
            print(f"Error sumary_size '{col}': {str(err)}")


    def reduce_object(self, data_serie: pd.Series) -> pd.Series:
        """
        Reduce el consumo de memoria de una columna de tipo objeto (string) convirtiéndola 
        a tipo entero.

        Args:
            data_serie (pd.Series): Columna de tipo objeto a procesar.

        Returns:
            pd.Series: Columna de tipo entero con el consumo de memoria reducido.
        """

        try:
            self.before_size = round(sys.getsizeof(data_serie) / 1024 ** 2,2)
                    
            transformlabel = {v:k for k,v in enumerate(data_serie.unique())}
            
            data_serie = data_serie.map(transformlabel).astype('int8')

            self.after_size = round(sys.getsizeof(data_serie) / 1024 ** 2,2)
        
            return data_serie
        
        except Exception as err:
            print(f"Error reducing object column: {str(err)}")
            return False

    def reduce_float(self, data_serie: pd.Series) -> pd.Series:
        """
        Reduce el consumo de memoria de una columna de tipo float ajustando su tipo de dato.

        Args:
            data_serie (pd.Series): Columna de tipo float a procesar.

        Returns:
            pd.Series: Columna con el tipo de dato ajustado y el consumo de memoria reducido.
        """
        try:
            self.before_size = round(sys.getsizeof(data_serie) / 1024 ** 2,2)
                    
            min_value, max_value = data_serie.min(), data_serie.max()
            
            if min_value >= np.finfo('float16').min and max_value <= np.finfo('float16').max:
                data_serie = data_serie.astype('float16')
            elif min_value >= np.finfo('float32').min and max_value <= np.finfo('float32').max:
                data_serie = data_serie.astype('float32')
            else:
                data_serie = data_serie.astype('float64')
                
            self.after_size = round(sys.getsizeof(data_serie) / 1024**2,2)

            return data_serie
        
        except Exception as err:
            print(f"Error reducing float column: {str(err)}")
            return False

    def reduce_int(self, data_serie: pd.Series) -> pd.Series:
        """
        Reduce el consumo de memoria de una columna de tipo entero ajustando su tipo de dato.

        Args:
            data_serie (pd.Series): Columna de tipo entero a procesar.

        Returns:
            pd.Series: Columna con el tipo de dato ajustado y el consumo de memoria reducido.
        """
        try:
            self.before_size = round(sys.getsizeof(data_serie) / 1024 ** 2,2)
                    
            min_value, max_value = data_serie.min(), data_serie.max()
            
            if min_value >= np.iinfo('int8').min and max_value <= np.iinfo('int8').max:
                data_serie = data_serie.astype('int8')
            elif min_value >= np.iinfo('int16').min and max_value <= np.iinfo('int16').max:
                data_serie = data_serie.astype('int16')
            elif min_value >= np.iinfo('int32').min and max_value <= np.iinfo('int32').max:
                data_serie = data_serie.astype('int32')
            else:
                data_serie = data_serie.astype('int64')
                
            self.after_size = round(sys.getsizeof(data_serie) / 1024**2,2)

            return data_serie
        
        except Exception as err:
            print(f"Error reducing int column: {str(err)}")
            return data_serie  

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


def cambiar_nombres_columnas(df, **kwargs):
    """
    Cambia los nombres de las columnas de un DataFrame.

    Parámetros de entrada:
        - df: DataFrame. El dataframe en el que se cambiarán los nombres de las columnas.
        - **kwargs: Diccionario de argumentos clave-valor donde la clave representa el nombre actual de la columna
                    y el valor representa el nuevo nombre de la columna.

    Retorna:
        DataFrame. El dataframe con los nombres de las columnas modificados.

    Ejemplo:
        df = cambiar_nombres_columnas(df, columna1='nueva_columna1', columna2='nueva_columna2')
    """
    try:
        
        for columna_actual in kwargs.keys():
            if columna_actual not in df.columns:
                raise ValueError(f"La columna '{columna_actual}' no existe en el DataFrame.")

        
        df = df.rename(columns=kwargs)

    except Exception as e:
        print(f"Error al cambiar los nombres de las columnas: {e}")

    return df


def comprobacion_outliers(dataframe: pd.DataFrame, nombre_columna: str) -> dict:
    '''
    Esta función calcula el número de outliers y su proporción con respecto al total en una columna numérica de un DataFrame de Pandas.

    Args:
    - dataframe: DataFrame de Pandas que contiene los datos.
    - nombre_columna: Nombre de la columna para la cual se desea detectar los outliers. Se deberá indicar en formato string.

    Return:
    - Diccionario con el número de outliers en la columna especificada y el porcentaje de outliers en relación al total de datos.
    '''
    try:
        if not isinstance(nombre_columna, str):
            raise TypeError("El nombre de la columna debe ser un string.")
            
        df = dataframe[nombre_columna]
        q1 = np.percentile(df, 25)
        q3 = np.percentile(df, 75)
        rango_intercuartilico = q3 - q1 
        outliers = df[(df < (q1 - 1.5 * rango_intercuartilico)) | (df > (q3 + 1.5 * rango_intercuartilico))]
        num_outliers = len(outliers)
        porcentaje_outliers = round((num_outliers / len(df)) * 100, 2)
        
        result = {
            "numero_outliers": num_outliers,
            "porcentaje_outliers": porcentaje_outliers
        }
        
        return result

    except KeyError:
        raise KeyError("Error: La columna especificada no existe en el DataFrame.")
    except TypeError as e:
        raise TypeError("Error: " + str(e))
    except Exception as e:
        raise Exception("Error: Se produjo un problema al procesar la función. Por favor, revisa la documentación de la función, verifica que los parámetros de entrada estén correctamente indicados y revisa los datos de tu DataFrame.")


def create_dataframe_yahoo_finance(symbol):
    """
    Crea un DataFrame a partir de los datos descargados de Yahoo Finance para un símbolo específico.

    Parámetros de entrada:
        - symbol: str. El símbolo del activo financiero para el cual se desea obtener los datos.

    Retorna:
        DataFrame. El dataframe con los datos descargados de Yahoo Finance.

    """
    try:
        data = yf.download(symbol)
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error al descargar los datos de Yahoo Finance para el símbolo {symbol}: {str(e)}")
        return None


def eliminar_unidades_metricas(df:pd.DataFrame, columna:str):
    """
    Elimina las unidades métricas de una columna de un DataFrame y la convierte a tipo float.

    Args:
        df (pandas.DataFrame): El DataFrame que contiene la columna con unidades métricas.
        columna (str): El nombre de la columna a procesar.

    Returns:
        pandas.DataFrame: El DataFrame modificado con la unidad métrica eliminada y la columna convertida a tipo float.
    """
    try:
        valores = df[columna].astype(str)
        valores_sin_unidades = valores.apply(lambda x: re.sub(r'[a-zA-Z]+', '', x))
        valores_sin_unidades = valores_sin_unidades.str.strip()
        df[columna] = valores_sin_unidades.astype(float)
        return df
    except Exception as e:
        print(f"Error: {str(e)}")


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


    if target not in dataframe.columns:
        print("La columna target no existe en el DataFrame.")
        return None

    if columna_categorica not in dataframe.columns:
        print("La columna columna_categorica no existe en el DataFrame.")
        return None

    if dataframe[target].nunique() != 2:
        print("La columna target no es binaria.")
        return None

    try:
        dict_proporcional = dict(dataframe.groupby(columna_categorica)[target].mean())
        dataframe[nueva_columna] = dataframe[columna_categorica].map(dict_proporcional)
        return dataframe
    except (KeyError, TypeError) as e:
        print("Ocurrió un error al codificar la columna categórica:", str(e))
        return None


def eliminacion_outliers(dataframe: pd.DataFrame, nombre_columna: str):
    '''
    Esta función elimina las filas del DataFrame que contienen valores atípicos (outliers) en una columna especificada.

    Args:
    - dataframe: DataFrame de Pandas que contiene los datos.
    - nombre_columna: Nombre de la columna en la cual se desean eliminar las filas con outliers. Se deberá indicar en formato string.

    Return:
    - Devuelve el DataFrame sin los valores atípicos de la columna especificada.
    '''

    if not isinstance(nombre_columna, str):
        raise TypeError("El nombre de la columna debe ser un string.")

    if nombre_columna not in dataframe.columns:
        raise KeyError("La columna especificada no existe en el DataFrame.")

    df = dataframe.copy()
    q1 = np.percentile(df[nombre_columna], 25)
    q3 = np.percentile(df[nombre_columna], 75)
    rango_intercuartilico = q3 - q1
    df = df[(df[nombre_columna] >= (q1 - 1.5 * rango_intercuartilico)) & (df[nombre_columna] <= (q3 + 1.5 * rango_intercuartilico))]

    return df


def leer_csv_desde_zip(ruta_archivo: str, nombre_archivo_csv: str)-> pd.DataFrame:
    """
    Lee un archivo CSV desde un archivo .zip.

    Args:
        ruta_archivo (str): La ruta del archivo .zip.
        nombre_archivo_csv (str): El nombre del archivo CSV que se desea leer desde el .zip.

    Returns:
        pd.DataFrame: Los datos leídos del archivo CSV como un DataFrame de pandas.
    """

    try:
        archivo_zip = zipfile.ZipFile(ruta_archivo)  # Abrir el archivo .zip
        contenido_zip = archivo_zip.namelist()  # Leer el contenido del archivo .zip
        if nombre_archivo_csv in contenido_zip:  # Verificar si el archivo CSV está presente en el .zip
            with archivo_zip.open(nombre_archivo_csv) as archivo_csv:  # Leer el archivo CSV
                datos = pd.read_csv(archivo_csv)
            return datos
        else:
            print("El archivo CSV no se encuentra en el archivo .zip.")
            return None
    except FileNotFoundError:
        print("El archivo .zip no fue encontrado.")
        return None
    except Exception as e:
        print("Ocurrió un error al leer el archivo .zip:", str(e))
        return None


def leer_csv_desde_rar(ruta_archivo:str, nombre_archivo_csv:str) -> pd.DataFrame:
    
    """
        Lee un archivo CSV desde un archivo .rar.

    Args:
        ruta_archivo (str): La ruta del archivo .rar.
        nombre_archivo_csv (str): El nombre del archivo CSV que se desea leer desde el .rar.

    Returns:
        pd.DataFrame: Los datos leídos del archivo CSV como un DataFrame de pandas.

    """
     
    try:
        archivo_rar = rarfile.RarFile(ruta_archivo) # Abrir el archivo .rar
        contenido_rar = archivo_rar.namelist() # Leer el contenido del archivo .rar
        if nombre_archivo_csv in contenido_rar: # Verificar si el archivo CSV está presente en el .rar          
            with archivo_rar.open(nombre_archivo_csv) as archivo_csv: # Leer el archivo CSV
                datos = pd.read_csv(archivo_csv)
                return datos
        else:
            print("El archivo CSV no se encuentra en el archivo .rar.")
            return None
    except FileNotFoundError:
        print("El archivo .rar no fue encontrado.")
        return None
    except Exception as e:
        print("Ocurrió un error al leer el archivo .rar:", str(e))
        return None


def limpiar_columnas_numericas(dataframe, columna, caracteres_especiales, valor_reemplazo):
    """
    Limpia una columna de un DataFrame, reemplazando los caracteres especiales inválidos
    por un valor de reemplazo dado.

    Args:
        dataframe (pandas.DataFrame): El DataFrame que contiene la columna a limpiar.
        columna (str): El nombre de la columna a limpiar.
        caracteres_especiales (list): Lista de caracteres especiales inválidos.
        valor_reemplazo (str): Valor utilizado para reemplazar los caracteres especiales.

    Returns:
        pandas.DataFrame: El DataFrame con la columna limpia.

    Raises:
        Exception: Si se encuentra un caracter especial inválido en la columna.
    """
   
    columna_datos = dataframe[columna]
    
    for i, dato in enumerate(columna_datos):
        try:
            for caracter in caracteres_especiales:
                if caracter in dato:
                    raise Exception("Se encontró un caracter especial inválido en la columna.")
            
            for caracter in caracteres_especiales:
                dato = dato.replace(caracter, valor_reemplazo)
            
            columna_datos[i] = dato
        
        except Exception as e:
            print(f"Error: {str(e)}")
    
    dataframe[columna] = columna_datos
    
    return dataframe


def obtener_hora_minuto_segundo(df:pd.DataFrame, columna_hora:str):
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


def split_and_encode_strings(column:pd.Series, use_encoding: bool = False ) -> pd.DataFrame:
    """
    Separa una columna de un DataFrame utilizando cualquier carácter que no sea una letra o un número como separador,
    y opcionalmente aplica one-hot encoding (get dummies) a las palabras separadas.

    Args:
        column (pd.Series): La columna del DataFrame que se desea separar y encodear.
        use_encoding (bool, optional): Indica si se debe aplicar one-hot encoding a las palabras separadas.
            Por defecto es False.

    Returns:
        pd.DataFrame: Un DataFrame con las palabras separadas en una sola columna si use_encoding es False,
            o un DataFrame con columnas correspondientes a cada palabra si use_encoding es True.

    """

    try:
        separador = re.compile(r'[^a-zA-Z0-9]+')
        palabras = column.apply(lambda x: separador.split(x))
        if use_encoding:
            df = pd.get_dummies(palabras.apply(pd.Series).stack()).groupby(level=1).sum()
        else:
            df = pd.DataFrame(palabras)
        return df
    except Exception as e:
        print("Ocurrió un error al separar y encodear las strings:", str(e))
        return None


def tratar_valores_nulos(dataframe:pd.DataFrame, opcion:str) -> Union[pd.DataFrame, None]:
    """
    Trata los valores nulos en un DataFrame según la opción seleccionada.

    Args:
        dataframe (pd.DataFrame): El DataFrame que se desea procesar.
        opcion (str): La opción seleccionada para tratar los valores nulos.
            Opciones disponibles: 'eliminar', 'rellenar_cero', 'rellenar_media'.

    Returns:
        Union[pd.DataFrame, None]: El DataFrame con los valores nulos tratados según la opción seleccionada.
        En caso de error, retorna None.

    """
    try:
        if opcion == 'eliminar':
            dataframe_sin_nulos = dataframe.dropna()
            return dataframe_sin_nulos
        elif opcion == 'rellenar_cero':
            dataframe_rellenado = dataframe.fillna(0)
            return dataframe_rellenado
        elif opcion == 'rellenar_media':
            dataframe_rellenado = dataframe.fillna(dataframe.mean())
            return dataframe_rellenado
        else:
            print("Opción inválida. Las opciones disponibles son: 'eliminar', 'rellenar_cero', 'rellenar_media'.")
            return dataframe
    except Exception as e:
        print("Ocurrió un error al tratar los valores nulos:", str(e))
        return None



