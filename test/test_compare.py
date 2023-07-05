import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from src.modeling import comparar_modelos



@pytest.fixture
def modelos_entrenados():
    # Entrenar modelos de prueba
    X, y = make_regression(n_samples=100, n_features=3, random_state=42)
    modelo1 = LinearRegression()
    modelo1.fit(X, y)
    modelo2 = RandomForestRegressor()
    modelo2.fit(X, y)
    return [modelo1, modelo2]

@pytest.fixture
def datos_prueba():
    # Generar datos de prueba
    X_test, y_test = make_regression(n_samples=50, n_features=3, random_state=42)
    return X_test, y_test

def test_comparar_modelos(modelos_entrenados, datos_prueba):
    modelos = modelos_entrenados
    X_test, y_test = datos_prueba

    # Comparar modelos
    mejor_modelo = comparar_modelos(modelos, X_test, y_test)

    # Verificar el resultado
    assert mejor_modelo is not None
    assert isinstance(mejor_modelo, LinearRegression) or isinstance(mejor_modelo, RandomForestRegressor)