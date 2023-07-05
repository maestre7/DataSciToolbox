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
    
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def train_decision_tree(X_train, y_train, max_depth=None, min_samples_split=2):
    """
    Función para entrenar decision tree.
    
    Args:
        X_train (array-like): Matriz de características de entrenamiento.
        y_train (array-like): Vector de etiquetas de entrenamiento.
        max_depth (int or None, optional): La profundidad máxima del árbol. 
                                           Si es None, se expande hasta que todas las hojas sean puras o hasta que 
                                           todas las hojas contengan menos de min_samples_split muestras.
        min_samples_split (int, optional): El número mínimo de muestras requeridas para dividir un nodo interno.
    
    Returns:
        model: Modelo decision tree entrenado.
    """
    try:
        model_dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
        model_dt.fit(X_train, y_train)
        
        return model_dt
    except Exception as e:
        print(f"Error en el entrenamiento: {e}")
        return None


def train_knn(X_train, y_train, n_neighbors=5, scale_features=True):
    """
    Función para entrenar knn.
    
    Args:
        X_train (array-like): Matriz de características de entrenamiento.
        y_train (array-like): Vector de etiquetas de entrenamiento.
        n_neighbors (int, optional): El número de vecinos a tener en cuenta durante la clasificación.
        scale_features (bool, optional): Indica si se escala la matriz de características. 
                                         Por defecto, es True.
    
    Returns:
        model_knn: Modelo knn entrenado.
    """
    try:
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        
        model_knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        model_knn.fit(X_train, y_train)
        
        return model_knn
    except Exception as e:
        print(f"Error en el entrenamiento: {e}")
        return None


def train_linear_regression(X_train, y_train, fit_intercept=True, scale_features=True):
    """
    Función para entrenar linear regression.
    
    Args:
        X_train (array-like): Matriz de características de entrenamiento.
        y_train (array-like): Vector de etiquetas de entrenamiento.
        fit_intercept (bool, optional): Indica si se ajusta el término de intercepción. 
                                        Por defecto, es True.
        scale_features (bool, optional): Indica si se escala la matriz de características. 
                                         Por defecto, es True.
    
    Returns:
        model_lr: Modelo linear regression entrenado.
    """
    try:
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        
        model_lr = LinearRegression(fit_intercept=fit_intercept)
        model_lr.fit(X_train, y_train)
        
        return model_lr
    except Exception as e:
        print(f"Error en el entrenamiento: {e}")
        return None


def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Función para entrenar random forest.
    
    Args:
        X_train (array-like): Matriz de características de entrenamiento.
        y_train (array-like): Vector de etiquetas de entrenamiento.
        n_estimators (int, optional): El número de árboles en el bosque.
    
    Returns:
        model_rf: Modelo random forest entrenado.
    """
    try:
        model_rf = RandomForestRegressor(n_estimators=n_estimators)
        model_rf.fit(X_train, y_train)
        
        return model_rf
    except Exception as e:
        print(f"Error en el entrenamiento: {e}")
        return None


def train_svr(X_train, y_train, kernel='rbf', scale_features=True):
    """
    Función para entrenar SVR.
    
    Args:
        X_train (array-like): Matriz de características de entrenamiento.
        y_train (array-like): Vector de etiquetas de entrenamiento.
        kernel (str, optional): El kernel a utilizar. Puede ser 'linear', 'poly', 'rbf', 'sigmoid' o una función personalizada.
        scale_features (bool, optional): Indica si se escala la matriz de características. 
                                         Por defecto, es True.
    
    Returns:
        model_svr: Modelo SVR entrenado.
    """
    try:
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        
        model_svr = SVR(kernel=kernel)
        model_svr.fit(X_train, y_train)
        
        return model_svr
    except Exception as e:
        print(f"Error en el entrenamiento: {e}")
        return None