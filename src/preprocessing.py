import pandas as pd
import numpy as np
import os

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
