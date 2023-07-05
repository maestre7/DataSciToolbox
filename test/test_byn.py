import pytest
import os
from PIL import Image

# Importar la función a testear
from src.modeling import ByN

# Definir la ruta de la imagen de prueba
TEST_IMAGE_PATH = "test/img/Cocodrilo.jpg"

# Definir la función de test
def test_ByN():
    # Comprobar si la imagen de prueba existe
    assert os.path.exists(TEST_IMAGE_PATH), f"La imagen de prueba no existe en la ruta especificada: {TEST_IMAGE_PATH}"

    # Llamar a la función a testear
    ByN(TEST_IMAGE_PATH)

    # Comprobar si se generó correctamente la imagen en blanco y negro
    image_path, image_extension = os.path.splitext(TEST_IMAGE_PATH)
    bn_image_path = f"{image_path}_bn{image_extension}"
    assert os.path.exists(bn_image_path), "No se generó la imagen en blanco y negro"

    # Comprobar si la imagen en blanco y negro tiene el modo correcto
    bn_image = Image.open(bn_image_path)
    assert bn_image.mode == "L", "La imagen en blanco y negro no tiene el modo correcto"

    # Eliminar la imagen en blanco y negro generada después de la prueba
    os.remove(bn_image_path)

# Ejecutar los tests
pytest.main([__file__])