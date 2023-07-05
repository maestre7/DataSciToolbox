from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler

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
    
