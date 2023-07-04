from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy, pandas
from typing import Any

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
        return None # type: ignore