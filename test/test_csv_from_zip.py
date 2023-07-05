import pytest
import pandas as pd
from src.preprocessing import leer_csv_desde_zip

def test_leer_csv_desde_zip():
    ruta_archivo = "csv_prueba_zip_rar.zip"
    nombre_archivo_csv = "csv_prueba_zip_rar.csv"

    dataframe = leer_csv_desde_zip(ruta_archivo, nombre_archivo_csv)
