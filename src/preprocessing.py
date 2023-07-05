import re
from typing import Union
import sys
import numpy as np
import pandas as pd
import rarfile
import zipfile


class ReduceMemory:
    """
    Clase para reducir el consumo de memoria de un DataFrame de Pandas
    """

    def __init__(self) -> None:
        self.before_size = 0
        self.after_size = 0

    def process(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce el consumo de memoria de un DataFrame de Pandas aplicando reducciones de 
        tipo de datos en cada columna del DataFrame.

        Args:
            data_df (pd.DataFrame): DataFrame de Pandas a procesar.

        Returns:
            pd.DataFrame: DataFrame de Pandas con el consumo de memoria reducido.
        """
        cols = data_df.columns

        for col in cols:
            try:
                dtype = data_df[col].dtype

                if dtype == 'object':
                    data_df[col] = self.reduce_object(data_df[col])
                elif dtype == 'float':
                    data_df[col] = self.reduce_float(data_df[col])
                elif dtype == 'int':
                    data_df[col] = self.reduce_int(data_df[col])
            except Exception as e:
                print(f"Error processing column '{col}': {str(e)}")

        return data_df

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
            return data_serie

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
            return data_serie

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
                    
            min_value,max_value = data_serie.min(), data_serie.max()
            
            if min_value >= np.iinfo('int8').min and max_value <= np.iinfo('int8').max:
                data_serie = data_serie.astype('int8')
            if min_value >= np.iinfo('int16').min and max_value <= np.iinfo('int16').max:
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

import pandas as pd


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

import pandas as pd
import yfinance as yf

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
