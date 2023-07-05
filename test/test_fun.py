from funcion import visualize_groupby
import pandas as pd
import matplotlib.pyplot as plt
from boundary import plot_decision_boundary
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Función de prueba
def test_plot_decision_boundary():
    # Generar datos de ejemplo
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=42)

    # Entrenar un modelo de regresión logística
    model = LogisticRegression()
    model.fit(X, y)

    # Visualizar los límites de decisión
    fig = plot_decision_boundary(model, X, y)

    # Verificar el tipo de objeto devuelto
    assert isinstance(fig, plt.Figure)



# --------------------------------
def test_visualize_groupby():
    # Crear un DataFrame de prueba
    data = {'A': ['foo', 'bar', 'foo', 'bar'], 'B': [1, 2, 3, 4]}
    df = pd.DataFrame(data)

    # Llamar a la función visualize_groupby con los argumentos apropiados
    grouped_data = visualize_groupby(df, 'A', 'B', 'sum')

    # Verificar que el número de filas es correcto
    assert len(grouped_data) == 2

    # Verificar que las columnas coinciden
    assert list(grouped_data.columns) == ['A', 'B']

    # Verificar que el resultado del groupby es correcto
    assert grouped_data['A'].tolist() == ['bar', 'foo']
    assert grouped_data['B'].tolist() == [6, 4]
    
    
    
    # -------------------------------------------------
    
from nuev import plot_learning_curve
import numpy as np

def test_plot_learning_curve():
    train_scores = np.array([0.8, 0.85, 0.9, 0.92, 0.94])
    test_scores = np.array([0.75, 0.78, 0.82, 0.85, 0.88])
    train_sizes = np.array([100, 200, 300, 400, 500])

    # Llamar a la función para generar la curva de aprendizaje
    plot_learning_curve(train_scores, test_scores, train_sizes)

if __name__ == "__main__":
    test_plot_learning_curve()