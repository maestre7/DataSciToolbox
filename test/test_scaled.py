from src.modeling import comparar_scaled
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def test_comparar_scaled():
    # Generar datos de prueba
    X, y = make_regression(n_samples=100, n_features=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el modelo
    modelo = LinearRegression()

    # Ejecutar la función comparar_scaled
    comparar_scaled(modelo, X_train, X_test, y_train, y_test)