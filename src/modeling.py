import os
import pickle

def export_import_model(model:object, path_model:str, name:str, save:bool=True, open:bool=False):
    '''
    Funcion para exportar o importar el modelo entrenado.

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

    if save == True:
        with open(filename, 'wb') as archivo_salida:
            pickle.dump(model, archivo_salida)

    if open == True:
        with open(filename, 'rb') as archivo_entrada:
         model_pretrained = pickle.load(archivo_entrada)