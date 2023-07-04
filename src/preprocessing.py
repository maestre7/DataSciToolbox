import zipfile
import pandas as pd

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