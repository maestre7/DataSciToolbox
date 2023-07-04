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

from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np

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
    mejor_r2 = -np.inf
    mejor_mae = np.inf
    mejor_mape = np.inf
    mejor_mse = np.inf
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        
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