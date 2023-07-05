import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y):
    """
    Genera una representación visual de los límites de decisión de un modelo de clasificación.

    Args:
        model: Modelo de clasificación entrenado.
        X: Datos de entrada.
        y: Etiquetas correspondientes.

    Returns:
        object: Objeto de la figura del gráfico generado.
    """
    try:
        # Crear una malla de puntos para evaluar los límites de decisión
        h = 0.02  # Tamaño del paso en la malla
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtener las predicciones para los puntos de la malla
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Crear el gráfico de los límites de decisión y los puntos de datos
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')

        # Devolver el objeto de la figura para su posterior verificación
        return plt.gcf()

    except Exception as e:
        print(f"Error en la función plot_decision_boundary: {e}")
        return None

# ----------------------------------------------------------------------

def visualize_groupby(data, group_col, agg_col, agg_func):
    """
    Realiza un groupby en un DataFrame y visualiza los resultados.

    Args:
        data: DataFrame de entrada.
        group_col: Columna utilizada para realizar el groupby.
        agg_col: Columna utilizada para realizar la agregación.
        agg_func: Función de agregación a aplicar.

    Returns:
        DataFrame: Datos agrupados.
    """
    try:
        # Realizar el groupby
        grouped_data = data.groupby(group_col)[agg_col].agg(agg_func).reset_index()

        # Crear el gráfico de barras
        fig, ax = plt.subplots()
        ax.bar(grouped_data[group_col], grouped_data[agg_col])
        ax.set_xlabel(group_col)
        ax.set_ylabel(agg_col)

        return grouped_data

    except Exception as e:
        print(f"Error en la función visualize_groupby: {e}")
        return None



  
# --------------------------------------------------------------------------------
import matplotlib.pyplot as plt

def plot_learning_curve(train_scores, test_scores, train_sizes):
    """
    Genera una curva de aprendizaje que muestra el rendimiento del modelo en función del tamaño del conjunto de entrenamiento.

    Args:
        train_scores: Puntuaciones de entrenamiento.
        test_scores: Puntuaciones de prueba.
        train_sizes: Tamaños del conjunto de entrenamiento.

    Returns:
        None
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_scores, label='Train')
        plt.plot(train_sizes, test_scores, label='Test')
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error en la función plot_learning_curve: {e}")


