import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Any, List
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve
from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.metrics import roc_curve


def plot_scatter_with_reference(y_test: np.array, predictions: np.array, title: str) -> None:
    """
    Genera un gráfico de dispersión con una línea de referencia para comparar los valores de prueba con los valores predichos.

    Args:
        y_test (array-like): Valores de prueba.
        predictions (array-like): Valores predichos.
        title (str): Título del gráfico.

    Returns:
        None
    """
    try:
        sns.set(style="darkgrid")
        plt.scatter(y_test, predictions, color='blue',
                    label='Valores de prueba vs. Valores predichos')
        # Línea de referencia: valores reales = valores predichos
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.scatter(y_test, y_test, color='red', label='Valores de prueba')
        plt.xlabel('Valores de prueba')
        plt.ylabel('Valores predichos')
        plt.title(title)
        plt.legend()
        plt.show()
    except Exception as e:
        print("Ocurrió un error al generar el gráfico de dispersión:", str(e))
    finally:
        return None


def plot_learning_curve(estimator, X, y, cv=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Genera una curva de aprendizaje para un estimador dado.

    Parámetros:
    estimator: objeto de estimador
    X: matriz de características
    y: vector de etiquetas
    cv: validación cruzada (opcional)
    train_sizes: tamaños de los conjuntos de entrenamiento (opcional)

    Devuelve:
    None
    """
    try:
        plt.figure()
        plt.title("Curva de aprendizaje")
        plt.xlabel("Tamaño del conjunto de entrenamiento")
        plt.ylabel("Score")

        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes)

        # Calcular la media y el desvío estándar del training score
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)

        # Graficar la curva de aprendizaje del training score
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")

        plt.legend(loc="best")
    except Exception as e:
        print(f"An error occurred: {e}")


def plot_precision_recall_curve(y_true, y_prob):
    """
    Genera una curva de precisión-recall dadas las etiquetas verdaderas y las probabilidades predichas.

    Parámetros:
    y_true: vector de etiquetas verdaderas
    y_prob: vector de probabilidades predichas
    """
    try:
        # Verificar si los argumentos son válidos
        y_true = column_or_1d(y_true)
        y_prob = column_or_1d(y_prob)
        check_consistent_length(y_true, y_prob)

        # Calcular la precisión y el recall para diferentes umbrales
        precision, recall, _ = precision_recall_curve(y_true, y_prob)

        # Generar la gráfica
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


def plot_roc_curve(y_true, y_score):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
    except Exception as e:
        print(f'Error: {e}')
