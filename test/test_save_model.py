import os
import pickle
import warnings
from modeling import save_model


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