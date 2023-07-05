from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, silhouette_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from sklearn.utils import shuffle
from PIL import Image
import os
import pickle
import warnings


def balance_target_column_random(df, target_column):
    """
    Equilibra una columna objetivo especificada en el DataFrame utilizando sobremuestreo y submuestreo aleatorio.
    Los datos se mezclan antes de devolver el DataFrame equilibrado.

    Parámetros:
        df (pandas.DataFrame): El DataFrame de entrada que contiene la columna objetivo.
        target_column (str): El nombre de la columna objetivo a equilibrar.

    Retorna:
        pandas.DataFrame: Un nuevo DataFrame con la columna objetivo equilibrada y mezclada.

    """
    try:
        # Separar características (X) y variable objetivo (y)
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Instanciar RandomOverSampler y RandomUnderSampler
        oversampler = RandomOverSampler(random_state=42)
        undersampler = RandomUnderSampler(random_state=42)

        # Sobremuestrear la clase mayoritaria
        X_oversampled, y_oversampled = oversampler.fit_resample(X, y)

        # Submuestrear la clase minoritaria
        X_resampled, y_resampled = undersampler.fit_resample(X_oversampled, y_oversampled)

        # Crear un nuevo DataFrame equilibrado
        balanced_df = pd.concat([X_resampled, y_resampled], axis=1)

        # Mezclar los datos
        balanced_df = shuffle(balanced_df, random_state=42)

        return balanced_df

    except Exception as e:
        print("Error al equilibrar la columna objetivo:", str(e))
        return None
    

def balance_target_column_smote(df, target_column):
    """
    Equilibra una columna objetivo especificada en el DataFrame utilizando una combinación de sobremuestreo y submuestreo.
    Los datos se mezclan antes de devolver el DataFrame equilibrado.

    Parámetros:
        df (pandas.DataFrame): El DataFrame de entrada que contiene la columna objetivo.
        target_column (str): El nombre de la columna objetivo a equilibrar.

    Retorna:
        pandas.DataFrame: Un nuevo DataFrame con la columna objetivo equilibrada y mezclada.

    """
    try:
        # Separar características (X) y variable objetivo (y)
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Instanciar el muestreador SMOTEENN
        sampler = SMOTEENN(random_state=42)

        # Remuestrear los datos
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Crear un nuevo DataFrame equilibrado
        balanced_df = pd.concat([X_resampled, y_resampled], axis=1)

        # Mezclar los datos
        balanced_df = shuffle(balanced_df, random_state=42)

        return balanced_df

    except Exception as e:
        print("Error al equilibrar la columna objetivo:", str(e))
        return None
    
def ByN(path_imagen:str):

    """
    Función que devuelve la misma imagen que se introduce pero en blanco y negro.

    Args:
        path_imagen (str): La dirección de la imagen.
        
    Returns:
        None

    """
    try:

        imagen = Image.open(path_imagen)

        
        imagen = imagen.convert('L')
        imagen_2 = np.asarray(imagen,dtype=np.float)

        plt.figure(figsize=(8,8))
        plt.imshow(imagen_2,cmap='gray')
        plt.axis('off')
        plt.show()

    except Exception as e:
        error_message = "Fallo: " + str(e)
        print(error_message)

    return None

def comparar_modelos(modelos, X_test, y_test):
    """
    Funcion para comparar las metricas de varios modelos y devolver el modelo con mejores metricas.

    Parametros
    ----------
        modelos: lista con los modelos a comparar.

        X_test: variables a predecir.

        y_test: predicción.
        
    Returns
    -------
        Mejor modelo: el modelo con mejores métricas.
    """
    mejor_modelo = None
    mejor_r2 = -np.inf  #marcamos peor metrica posible
    mejor_mae = np.inf
    mejor_mape = np.inf
    mejor_mse = np.inf
    

        
    for modelo in modelos:
        try:
            y_pred = modelo.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            if r2 > mejor_r2:
                mejor_modelo = modelo
                mejor_r2 = r2
                mejor_mae = mae
                mejor_mape = mape
                mejor_mse = mse
        except Exception as e:
            print(f"Error al evaluar el modelo {modelo}: {str(e)}")
    
        print("Mejor modelo:", str(mejor_modelo))
        print("Mejor R2 Score:", mejor_r2)
        print("Mejor MAE:", mejor_mae)
        print("Mejor MAPE:", mejor_mape)
        print("Mejor MSE:", mejor_mse)
        
        return mejor_modelo
    

def comparar_scaled(modelo, x_train, x_test, y_train, y_test):
    '''
    Funcion para comparar las metricas de un modelo, escalando o no los datos.

    Parametros
    ----------
        modelo (objeto): El modelo de machine learning que se utilizará para las predicciones.

        x_train (array-like): Los datos de entrenamiento sin escalar.

        x_test (array-like): Los datos de prueba sin escalar.

        y_train (array-like): Las etiquetas de entrenamiento correspondientes a x_train.
        
        y_test (array-like): Las etiquetas de prueba correspondientes a x_test.
    '''
    # Entrenar el modelo con x_train sin escalar
    model1 = modelo
    try:
        model1.fit(x_train, y_train)
    except Exception as e:
        warnings.warn(f"Error al entrenar el modelo sin escalar: {str(e)}")

    # Obtener las predicciones del modelo en x_test sin escalar
    try:
        y_pred = model1.predict(x_test)
    except Exception as e:
        warnings.warn(f"Error al hacer predicciones sin escalar: {str(e)}")
        

    # Calcular las métricas de evaluación utilizando y_test y las predicciones correspondientes
    try:
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
    except Exception as e:
        warnings.warn(f"Error al calcular las métricas sin escalar: {str(e)}")
        r2, mae, mape, mse = None, None, None, None

    # Imprimir los resultados para x_train sin escalar
    print("Modelo: x_train")
    print("R2 Score:", r2)
    print("MAE:", mae)
    print("MAPE:", mape)
    print("MSE:", mse)
    print()

    # Crear el objeto StandardScaler
    scaler = StandardScaler()

    # Ajustar el scaler utilizando x_train
    try:
        scaler.fit(x_train)
    except Exception as e:
        warnings.warn(f"Error al ajustar el scaler: {str(e)}")

    # Escalar x_train y x_test
    try:
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
    except Exception as e:
        warnings.warn(f"Error al escalar los datos: {str(e)}")
        x_train_scaled, x_test_scaled = None, None

    # Entrenar el modelo con x_train_scaled
    model_scal = modelo
    try:
        model_scal.fit(x_train_scaled, y_train)
    except Exception as e:
        warnings.warn(f"Error al entrenar el modelo escalado: {str(e)}")

    # Obtener las predicciones del modelo en x_test_scaled
    try:
        y_pred_scaled = model_scal.predict(x_test_scaled)
    except Exception as e:
        warnings.warn(f"Error al hacer predicciones escaladas: {str(e)}")
        
    # Calcular las métricas de evaluación utilizando y_test y las predicciones correspondientes
    try:
        r2_scaled = r2_score(y_test, y_pred_scaled)
        mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
        mape_scaled = mean_absolute_percentage_error(y_test, y_pred_scaled)
        mse_scaled = mean_squared_error(y_test, y_pred_scaled)
    except Exception as e:
        warnings.warn(f"Error al calcular las métricas sin escalar: {str(e)}")
        r2_scaled, mae_scaled, mape_scaled, mse_scaled = None, None, None, None
    

    # Imprimir los resultados para x_train_scaled
    print("Modelo: x_train_scaled")
    print("R2 Score:", r2_scaled)
    print("MAE:", mae_scaled)
    print("MAPE:", mape_scaled)
    print("MSE:", mse_scaled)
    print()


def evaluacion_clas(nom_modelo: str, modelo: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, redondeo: int = None) -> pd.DataFrame: # type: ignore
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
        
        if len(np.unique(y_test)) > 2:
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
            accuracy = np.round(accuracy, redondeo)
            precision = np.round(precision, redondeo)
            recall = np.round(recall, redondeo)
            f1 = np.round(f1, redondeo)
            roc_auc = np.round(roc_auc, redondeo)
        
        result_df = pd.DataFrame(data=[[nom_modelo, accuracy, precision, recall, f1, roc_auc]], 
                                columns=["Model", 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
        return result_df
    
    except Exception as e:
        print("Error al evaluar el modelo'{}':".format(nom_modelo))
        return None # type: ignore


def evaluacion_reg(nom_modelo:str, modelo:Any, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray) -> pd.DataFrame:  
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
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        y_test_mean= y_test.mean()
        mae_ratio= mae/y_test_mean
        rmse_ratio= rmse/y_test_mean
        
        result_df = pd.DataFrame(data=[[nom_modelo, mae, mse, rmse, r2, mae_ratio, rmse_ratio]], 
                                columns=["Model", 'MAE', 'MSE', 'RMSE', 'R2 Score', "MAE Ratio", "RMSE Ratio"])
        return result_df
    
    except Exception as e:
        print("Error al evaluar el modelo'{}':".format(nom_modelo))
        return None
    

def export_import_model(model, path_model, name, save=True, open=False):
    '''
    Funcion para exportar o importar el modelo entrenado
    Parametros
    ----------
        model: el modelo a guardar.
        path_model: directorio donde se almacenará.
        name: nombre del modelo a guardar.
        
        save: por defecto nos guarda el modelo entrenado.
        open: por defecto False. Si True, importamos el modelo entrenado
    '''

    # Exportamos el modelo con el nombre seleccionado, al path escogido.
    filename = os.path.join(path_model, name)

    if save:
        try:
            with open(filename, 'wb') as archivo_salida:
                pickle.dump(model, archivo_salida)
        except Exception as e:
            print(f"Error al guardar el modelo: {str(e)}")

    if open:
        try:
            with open(filename, 'rb') as archivo_entrada:
                model_pretrained = pickle.load(archivo_entrada)
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            model_pretrained = None

    return model_pretrained


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
    

def reductor_calidad(path_imagen:str,n:int):
    
    """
    Función para hacer reducir el tamaño de memoria que ocupa una imagen reduciendo el numero de colores de ésta. 

    Args:
        path_imagen (str): El data frame que se quiere modelar.
        n (int): Número de colores total del que se desea la foto.

    Returns:
        None

    """
    try:

        imagen = Image.open(path_imagen)
        imagen_1 = np.asarray(imagen,dtype=np.float32)/255

        w, h = imagen.size
        colors = imagen.getcolors(w * h)
        num_colores_0 = len(colors) 
        num_pixels_0 = w*h 
        
        R = imagen_1[:,:,0]
        G = imagen_1[:,:,1]
        B = imagen_1[:,:,2]
        XR = R.reshape((-1, 1))  
        XG = G.reshape((-1, 1)) 
        XB = B.reshape((-1, 1)) 
        X = np.concatenate((XR,XG,XB),axis=1)
        
        k_means = KMeans(n_clusters=n)
        k_means.fit(X)
        centroides = k_means.cluster_centers_
        etiquetas = k_means.labels_
        m = XR.shape
        for i in range(m[0]):
            XR[i] = centroides[etiquetas[i]][0] 
            XG[i] = centroides[etiquetas[i]][1] 
            XB[i] = centroides[etiquetas[i]][2] 
        XR.shape = R.shape 
        XG.shape = G.shape
        XB.shape = B.shape 
        XR = XR[:, :, np.newaxis]  
        XG = XG[:, :, np.newaxis]
        XB = XB[:, :, np.newaxis]
        Y = np.concatenate((XR,XG,XB),axis=2)
        
        plt.figure(figsize=(12,12))
        plt.imshow(Y)
        plt.axis('off')
        plt.show()

        print (u'Número de colores iniciales = ', num_colores_0)
        print (u'Número de colores finales = ', n)

    except Exception as e:
        error_message = "Fallo: " + str(e)
        print(error_message)

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
    

def train_test_split_df(df, target_col, test_percent, random_state):
    """
    Divide un DataFrame en conjuntos de entrenamiento y prueba para el aprendizaje automático.

    Parámetros:
        df (pandas.DataFrame): El DataFrame de entrada.
        target_col (str): El nombre de la columna objetivo en el DataFrame.
        test_percent (float): El porcentaje de datos para usar en la prueba (entre 0 y 1).
        random_state (int): El estado aleatorio para garantizar la reproducibilidad.

    Retorna:
        X_train (pandas.DataFrame): Las características del conjunto de entrenamiento.
        X_test (pandas.DataFrame): Las características del conjunto de prueba.
        y_train (pandas.Series): La variable objetivo del conjunto de entrenamiento.
        y_test (pandas.Series): La variable objetivo del conjunto de prueba.

    Imprime:
        La forma de cada conjunto para confirmar la dimensión.
    """
    try:
        # Separar características (X) y variable objetivo (y)
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=random_state)

        print("Forma de X:", X.shape)
        print("Forma de y:", y.shape)

        print("Forma de X_train:", X_train.shape)
        print("Forma de X_test:", X_test.shape)
        print("Forma de y_train:", y_train.shape)
        print("Forma de y_test:", y_test.shape)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print("Error al dividir los datos en conjuntos de entrenamiento y prueba:", str(e))
        return None, None, None, None
    

def comparar_scaled(modelo, x_train, x_test, y_train, y_test):
    '''
    Funcion para comparar las metricas de un modelo, escalando o no los datos.

    Parametros
    ----------
        modelo (objeto): El modelo de machine learning que se utilizará para las predicciones.

        x_train (array-like): Los datos de entrenamiento sin escalar.

        x_test (array-like): Los datos de prueba sin escalar.

        y_train (array-like): Las etiquetas de entrenamiento correspondientes a x_train.
        
        y_test (array-like): Las etiquetas de prueba correspondientes a x_test.
    '''
    # Entrenar el modelo con x_train sin escalar
    model1 = modelo
    try:
        model1.fit(x_train, y_train)
    except Exception as e:
        warnings.warn(f"Error al entrenar el modelo sin escalar: {str(e)}")

    # Obtener las predicciones del modelo en x_test sin escalar
    try:
        y_pred = model1.predict(x_test)
    except Exception as e:
        warnings.warn(f"Error al hacer predicciones sin escalar: {str(e)}")
        

    # Calcular las métricas de evaluación utilizando y_test y las predicciones correspondientes
    try:
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
    except Exception as e:
        warnings.warn(f"Error al calcular las métricas sin escalar: {str(e)}")
        r2, mae, mape, mse = None, None, None, None

    # Imprimir los resultados para x_train sin escalar
    print("Modelo: x_train")
    print("R2 Score:", r2)
    print("MAE:", mae)
    print("MAPE:", mape)
    print("MSE:", mse)
    print()

    # Crear el objeto StandardScaler
    scaler = StandardScaler()

    # Ajustar el scaler utilizando x_train
    try:
        scaler.fit(x_train)
    except Exception as e:
        warnings.warn(f"Error al ajustar el scaler: {str(e)}")

    # Escalar x_train y x_test
    try:
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
    except Exception as e:
        warnings.warn(f"Error al escalar los datos: {str(e)}")
        x_train_scaled, x_test_scaled = None, None

    # Entrenar el modelo con x_train_scaled
    model_scal = modelo
    try:
        model_scal.fit(x_train_scaled, y_train)
    except Exception as e:
        warnings.warn(f"Error al entrenar el modelo escalado: {str(e)}")

    # Obtener las predicciones del modelo en x_test_scaled
    try:
        y_pred_scaled = model_scal.predict(x_test_scaled)
    except Exception as e:
        warnings.warn(f"Error al hacer predicciones escaladas: {str(e)}")
        
    # Calcular las métricas de evaluación utilizando y_test y las predicciones correspondientes
    try:
        r2_scaled = r2_score(y_test, y_pred_scaled)
        mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
        mape_scaled = mean_absolute_percentage_error(y_test, y_pred_scaled)
        mse_scaled = mean_squared_error(y_test, y_pred_scaled)
    except Exception as e:
        warnings.warn(f"Error al calcular las métricas sin escalar: {str(e)}")
        r2_scaled, mae_scaled, mape_scaled, mse_scaled = None, None, None, None
    

    # Imprimir los resultados para x_train_scaled
    print("Modelo: x_train_scaled")
    print("R2 Score:", r2_scaled)
    print("MAE:", mae_scaled)
    print("MAPE:", mape_scaled)
    print("MSE:", mse_scaled)
    print()