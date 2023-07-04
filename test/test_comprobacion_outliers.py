import numpy as np
import pandas as pd
import seaborn as sns
from preprocessing import comprobacion_outliers

def test_comprobacion_outliers():
    # Crear un DataFrame de ejemplo
    data = {'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]}
    df = pd.DataFrame(data)
    
    # Llamar a la función y obtener los resultados
    result = comprobacion_outliers(df, 'A')
    
    # Verificar si se generó el gráfico correctamente
    assert isinstance(result, sns.axisgrid.BoxPlot)
    
    # Verificar el número de outliers y su porcentaje
    assert len(result.outliers) == 0
    assert result.outliers_percentage == 0.0