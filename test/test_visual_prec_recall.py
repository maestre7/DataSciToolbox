import pytest
from visualization import plot_precision_recall_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


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


# Generar datos de prueba binarios
X, y = make_classification(n_classes=2, random_state=0)

# Ajustar un modelo de regresión logística
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# Obtener las probabilidades predichas
y_prob = clf.predict_proba(X)[:, 1]

# Llamar a la función plot_precision_recall_curve fuera de la función de prueba
plot_precision_recall_curve(y, y_prob)
