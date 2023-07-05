import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston # type: ignore
from src.modeling import evaluacion_reg

def test_evaluacion_reg():
    # Cargar el dataset de ejemplo
    dataset = load_boston()
    X = dataset.data
    y = dataset.target

    # Dividir los datos en entrenamiento y evaluación
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear un modelo de regresión lineal
    modelo_lineal = LinearRegression()

    # Evaluar el modelo lineal utilizando la función evaluacion_reg
    result_df_lineal = evaluacion_reg("Linear Regression", modelo_lineal, X_train, y_train, X_test, y_test)

    # Verificar que el DataFrame tiene la estructura esperada para el modelo lineal
    assert isinstance(result_df_lineal, pd.DataFrame)
    assert result_df_lineal.shape == (1, 7)
    assert list(result_df_lineal.columns) == ["Model", 'MAE', 'MSE', 'RMSE', 'R2 Score', "MAE Ratio", "RMSE Ratio"]

    # Verificar que los valores de las métricas sean numéricos para el modelo lineal
    assert isinstance(result_df_lineal["MAE"].values[0], (int, float))
    assert isinstance(result_df_lineal["MSE"].values[0], (int, float))
    assert isinstance(result_df_lineal["RMSE"].values[0], (int, float))
    assert isinstance(result_df_lineal["R2 Score"].values[0], (int, float))
    assert isinstance(result_df_lineal["MAE Ratio"].values[0], (int, float))
    assert isinstance(result_df_lineal["RMSE Ratio"].values[0], (int, float))

    # Crear un modelo de regresión polinómica de grado 2
    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    modelo_polinomico = LinearRegression()

    # Evaluar el modelo polinómico utilizando la función evaluacion_reg
    result_df_polinomico = evaluacion_reg("Polynomial Regression (degree=2)", modelo_polinomico, X_train_poly, y_train, X_test_poly, y_test)

    # Verificar que el DataFrame tiene la estructura esperada para el modelo polinómico
    assert isinstance(result_df_polinomico, pd.DataFrame)
    assert result_df_polinomico.shape == (1, 7)
    assert list(result_df_polinomico.columns) == ["Model", 'MAE', 'MSE', 'RMSE', 'R2 Score', "MAE Ratio", "RMSE Ratio"]

    # Verificar que los valores de las métricas sean numéricos para el modelo polinómico
    assert isinstance(result_df_polinomico["MAE"].values[0], (int, float))
    assert isinstance(result_df_polinomico["MSE"].values[0], (int, float))
    assert isinstance(result_df_polinomico["RMSE"].values[0], (int, float))
    assert isinstance(result_df_polinomico["R2 Score"].values[0], (int, float))
    assert isinstance(result_df_polinomico["MAE Ratio"].values[0], (int, float))
    assert isinstance(result_df_polinomico["RMSE Ratio"].values[0], (int, float))

    # Verificar que los modelos se ajusten correctamente y generen predicciones
    assert result_df_lineal["Model"].values[0] == "Linear Regression"
    assert not np.isnan(result_df_lineal["MAE"].values[0])
    assert not np.isnan(result_df_lineal["MSE"].values[0])
    assert not np.isnan(result_df_lineal["RMSE"].values[0])
    assert not np.isnan(result_df_lineal["R2 Score"].values[0])
    assert not np.isnan(result_df_lineal["MAE Ratio"].values[0])
    assert not np.isnan(result_df_lineal["RMSE Ratio"].values[0])

    assert result_df_polinomico["Model"].values[0] == "Polynomial Regression (degree=2)"
    assert not np.isnan(result_df_polinomico["MAE"].values[0])
    assert not np.isnan(result_df_polinomico["MSE"].values[0])
    assert not np.isnan(result_df_polinomico["RMSE"].values[0])
    assert not np.isnan(result_df_polinomico["R2 Score"].values[0])
    assert not np.isnan(result_df_polinomico["MAE Ratio"].values[0])
    assert not np.isnan(result_df_polinomico["RMSE Ratio"].values[0])
