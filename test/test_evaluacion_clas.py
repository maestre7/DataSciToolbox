import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from modeling import evaluacion_clas

def test_evaluacion_clas_iris():
    # Cargar el conjunto de datos iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Dividir los datos en entrenamiento y evaluación
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = LogisticRegression()
    result = evaluacion_clas("Logistic Regression", modelo, X_train, y_train, X_test, y_test)
    assert result is not None
    assert result.shape == (1, 6)
    assert result["Model"].iloc[0] == "Logistic Regression"

    modelo = DecisionTreeClassifier()
    result = evaluacion_clas("Decision Tree", modelo, X_train, y_train, X_test, y_test)
    assert result is not None
    assert result.shape == (1, 6)
    assert result["Model"].iloc[0] == "Decision Tree"

    modelo = RandomForestClassifier()
    result = evaluacion_clas("Random Forest", modelo, X_train, y_train, X_test, y_test)
    assert result is not None
    assert result.shape == (1, 6)
    assert result["Model"].iloc[0] == "Random Forest"
