import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from src.modeling import train_decision_tree
from sklearn.tree import DecisionTreeRegressor
import pytest


def test_train_decision_tree():
    # Generar datos de ejemplo
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Llamar a la función para entrenar el modelo
    model = train_decision_tree(X_train, y_train, max_depth=3, min_samples_split=5)

    # Comprobar si el modelo es un objeto DecisionTreeRegressor
    assert isinstance(model, DecisionTreeRegressor)

    # Comprobar si el modelo ha sido entrenado correctamente
    assert model.tree_.max_depth == 3
    assert model.min_samples_split == 5

    # Comprobar si el modelo hace predicciones correctamente
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape