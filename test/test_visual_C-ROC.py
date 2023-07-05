import pytest
from visualization import plot_roc_curve


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


# Llamada a la función plot_roc_curve fuera de la función de prueba
y_true = [0, 0, 1, 1]
y_score = [0.1, 0.4, 0.35, 0.8]
plot_roc_curve(y_true, y_score)
