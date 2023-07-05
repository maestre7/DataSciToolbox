import pytest
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from src.modeling import reductor_calidad

def test_reductor_calidad():
    # Ruta de la imagen de prueba
    path_imagen = "test/img/Cocodrilo.jpg"
    # Número de colores deseados
    n = 64

    # Llamar a la función reductor_calidad
    reductor_calidad(path_imagen, n)

    # Asegurar que no se produjo ninguna excepción
    assert True


# Ejecutar el test
pytest.main([__file__, '-k', 'test_reductor_calidad'])