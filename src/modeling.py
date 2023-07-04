import os
import pickle
import warnings

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


import warnings
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        y_pred = []

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
        y_pred_scaled = []

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



