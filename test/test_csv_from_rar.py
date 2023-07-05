import pytest
import pandas as pd

from src.preprocessing import leer_csv_desde_rar

def test_leer_csv_desde_rar():
    ruta_archivo = "csv_prueba_zip_rar.rar"
    nombre_archivo_csv = "csv_prueba_zip_rar.csv"

    dataframe = leer_csv_desde_rar(ruta_archivo, nombre_archivo_csv)

