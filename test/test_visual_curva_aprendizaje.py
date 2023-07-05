import pytest
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from visualization import plot_learning_curve
import matplotlib.pyplot as plt


def test_plot_learning_curve():
    # Cargar un conjunto de datos de ejemplo
    digits = load_digits()
    X, y = digits.data, digits.target

    # Crear un objeto de estimador de ejemplo
    estimator = LogisticRegression(max_iter=10000)

    # Llamar a la función plot_learning_curve
    plot_learning_curve(estimator, X, y)

    # Verificar que se ha creado una figura
    fig = plt.gcf()
    assert fig is not None, "Expected a figure to be created, but no figure was found"

    # Verificar que la figura tiene un título
    title = fig.axes[0].get_title()
    assert title == "Curva de aprendizaje", f"Expected title 'Curva de aprendizaje', but got '{title}'"

    # Verificar que la figura tiene una etiqueta en el eje x
    xlabel = fig.axes[0].get_xlabel()
    assert xlabel == "Tamaño del conjunto de entrenamiento", f"Expected xlabel 'Tamaño del conjunto de entrenamiento', but got '{xlabel}'"

    # Verificar que la figura tiene una etiqueta en el eje y
    ylabel = fig.axes[0].get_ylabel()
    assert ylabel == "Score", f"Expected ylabel 'Score', but got '{ylabel}'"


# Cargar un conjunto de datos de ejemplo
digits = load_digits()
X, y = digits.data, digits.target

# Crear un objeto de estimador de ejemplo con max_iter=10000
estimator = LogisticRegression(max_iter=10000)

# Llamar a la función plot_learning_curve fuera de la función de prueba
plot_learning_curve(estimator, X, y)
