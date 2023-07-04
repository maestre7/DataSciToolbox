import os
import pickle
import pytest

from modeling import export_import_model

def test_export_import_model(tmpdir):
    # Crear un modelo de prueba
    model = "modelo de prueba"

    # Definir el directorio de prueba y el nombre de archivo
    path_model = str(tmpdir)
    name = "test_model.pkl"

    # Exportar el modelo
    export_import_model(model, path_model, name, save=True)

    # Comprobar si el archivo se ha creado correctamente
    assert os.path.isfile(os.path.join(path_model, name))

    # Importar el modelo
    loaded_model = export_import_model(None, path_model, name, open=True)

    # Comprobar si el modelo importado es igual al modelo original
    assert loaded_model == model

    # Intentar importar un archivo inexistente
    with pytest.raises(FileNotFoundError):
        export_import_model(None, path_model, "nonexistent.pkl", open=True)