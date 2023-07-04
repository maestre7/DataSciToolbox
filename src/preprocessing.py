import sys
import numpy as np
import pandas as pd
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
    except FileNotFoundError:
        print("El archivo .zip no fue encontrado.")
        return None
    except Exception as e:
        print("Ocurrió un error al leer el archivo .zip:", str(e))
        return None
