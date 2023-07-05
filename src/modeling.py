from typing import Any
import numpy, pandas

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluacion_clas(nom_modelo: str, modelo: Any, X_train: numpy.ndarray, y_train: numpy.ndarray, X_test: numpy.ndarray, y_test: numpy.ndarray, redondeo: int = None) -> pandas.DataFrame: # type: ignore
    """
    Función para evaluar las predicciones de un modelo de clasificación de machine learning, devolviendo diferentes métricas en un dataframe.
    En caso de tratarse de una clasificación multiclase, el average de las métricas será None.
    Args:
        nom_modelo (str): El nombre del modelo.
        modelo (Any): Modelo de machine learning para hacer las predicciones.
        X_train (numpy.ndarray): Variables predictivas de entrenamiento.
        y_train (numpy.ndarray): Target de entrenamiento.
        X_test (numpy.ndarray): Variables predictivas de evaluación.
        y_test (numpy.ndarray): Target de evaluación.
        redondeo (int): Cantidad de decimales para redondear las métricas. (default: None)

    Returns:
        pandas.DataFrame: Dataframe con todas las métricas de evaluación del modelo.
    """
    try:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_pred_prob = modelo.predict_proba(X_test)
        
        if len(numpy.unique(y_test)) > 2:
            average = None
            multi_class = 'ovr'
        else:
            average = 'binary'
            multi_class = 'raise'
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)
        roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class=multi_class)
        
        if redondeo is not None:
            accuracy = numpy.round(accuracy, redondeo)
            precision = numpy.round(precision, redondeo)
            recall = numpy.round(recall, redondeo)
            f1 = numpy.round(f1, redondeo)
            roc_auc = numpy.round(roc_auc, redondeo)
        
        result_df = pandas.DataFrame(data=[[nom_modelo, accuracy, precision, recall, f1, roc_auc]], 
                                columns=["Model", 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
        return result_df
    
    except Exception as e:
        print("Error al evaluar el modelo'{}':".format(nom_modelo))
        return None # type: ignore
    



def modelo_kmeans_df(data: pd.DataFrame, n:int):

    """
    Función para hacer un modelo de Clustering para machine learning, devolviendo diferentes métricas en un dataframe.
    Utiliza las columnas numericas del dataframe que se declare en los argumentos.
    Además del return genera una gáfica con la media de las distancias de k en el eje Y, y el numero de K en el eje X.


    Args:
        data (str): El data frame que se quiere modelar.
        n (int): Número máximo del rango de K que se quieren untilizar

    Returns:
        Un pandas.DataFrame con todas las métricas (silhouette_score, Average Distance y SSE (Sum of Squared Errors)) de evaluación del modelo.

    """
    try:
        rango = range(2, n)
        X = data.select_dtypes(include=np.number)

        k_values = []
        average_distances = []
        sse_values = []
        silhouette_scores = []

        for k in rango:
            modelo = KMeans(n_clusters=k, random_state=42)
            modelo.fit(X)

            distances = np.min(modelo.transform(X), axis=1)
            average_distance = np.mean(distances)
            average_distances.append(average_distance)

            sse = modelo.inertia_
            sse_values.append(sse)

            labels = modelo.labels_
            silhouette = silhouette_score(X, labels)
            silhouette_scores.append(silhouette)

            k_values.append(k)

        # Crear el DataFrame con las métricas
        df_metrics = pd.DataFrame({
            'K': k_values,
            'Average Distance': average_distances,
            'SSE': sse_values,
            'Silhouette Score': silhouette_scores
        })

        # Graficar la distancia media en función de K
        plt.plot(rango, average_distances, 'bo-')
        plt.xlabel('K')
        plt.ylabel('Average Distance')
        plt.title('K-Means Average Distance')
        plt.show()

    except Exception as e:
        error_message = "Fallo: " + str(e)
        print(error_message)

    return df_metrics

def evaluacion_reg(nom_modelo:str, modelo:Any, X_train:numpy.ndarray, y_train:numpy.ndarray, X_test:numpy.ndarray, y_test:numpy.ndarray) -> pandas.DataFrame:  
    '''
    Función para evaluar las predicciones de un modelo de regresión de machine learning, devolviendo diferentes métricas en un dataframe.

    Args:
        nom_modelo (str): El nombre del modelo.
        modelo (Any): Modelo de machine learning para hacer las predicciones.
        X_train (numpy.ndarray): Variables predictivas de entrenamiento.
        y_train (numpy.ndarray): Target de entrenamiento.
        X_test (numpy.ndarray): Variables predictivas de evaluación.
        y_test (numpy.ndarray): Target de evaluación.

    Returns:
        pandas.DataFrame: Dataframe con todas las métricas de evaluación del modelo.
    '''
    try:
        modelo.fit(X_train, y_train)
        
        y_pred= modelo.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = numpy.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        y_test_mean= y_test.mean()
        mae_ratio= mae/y_test_mean
        rmse_ratio= rmse/y_test_mean
        
        result_df = pandas.DataFrame(data=[[nom_modelo, mae, mse, rmse, r2, mae_ratio, rmse_ratio]], 
                                columns=["Model", 'MAE', 'MSE', 'RMSE', 'R2 Score', "MAE Ratio", "RMSE Ratio"])
        return result_df
    
    except Exception as e:
        print("Error al evaluar el modelo'{}':".format(nom_modelo))
        return None