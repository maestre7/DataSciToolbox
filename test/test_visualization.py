from ROC import plot_roc_curve
import pytest
from prec_recall import plot_precision_recall_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from curva_aprendizaje import plot_learning_curve
import matplotlib.pyplot as plt

# Precision Recall curva


def test_plot_precision_recall_curve():
    # Generar datos de prueba binarios
    X, y = make_classification(n_classes=2, random_state=0)

    # Ajustar un modelo de regresión logística
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    # Obtener las probabilidades predichas
    y_prob = clf.predict_proba(X)[:, 1]

    # Caso de prueba con datos válidos
    plot_precision_recall_curve(y, y_prob)

    # Caso de prueba con argumentos inválidos
    with pytest.raises(Exception):
        plot_precision_recall_curve(y, 'invalid')
        plot_precision_recall_curve('invalid', y_prob)
        plot_precision_recall_curve(y, [0.1, 0.4, 'invalid', 0.8])

    # Caso de prueba con argumentos de diferentes longitudes
    with pytest.raises(Exception):
        plot_precision_recall_curve(y, y_prob[:-1])
        plot_precision_recall_curve(y[:-1], y_prob)


# Curva de aprendizaje

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


# Plotear curva ROC


def test_plot_roc_curve():
    # Caso de prueba con datos válidos
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    plot_roc_curve(y_true, y_score)

    # Caso de prueba con datos inválidos
    with pytest.raises(Exception):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 'invalid', 0.8]
        plot_roc_curve(y_true, y_score)
