from Kmeans_df import modelo_kmeans_df
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from src.modeling import modelo_kmeans_df


def test_modelo_kmeans_df():
    
    np.random.seed(42)
    data = pd.DataFrame(np.random.rand(100, 5), columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])

    df_metrics = modelo_kmeans_df(data, 10)

    # Imprimir el resultado
    print("Resultados del modelo de K-Means:")
    print(df_metrics)


test_modelo_kmeans_df()




