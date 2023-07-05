import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from src.modeling import train_random_forest
from sklearn.ensemble import RandomForestRegressor
import pytest


def test_train_random_forest():
    # Generar datos de ejemplo
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Llamar a la función para entrenar el modelo
    model = train_random_forest(X_train, y_train, n_estimators=100)

    # Comprobar si el modelo es un objeto RandomForestRegressor
    assert isinstance(model, RandomForestRegressor)

    # Comprobar si el modelo ha sido entrenado correctamente
    assert model.n_estimators == 100

    # Comprobar si el modelo hace predicciones correctamente
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape