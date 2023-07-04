import pandas as pd
import plotly.graph_objects as go
import pytest
from src.visualization import plot_moving_averages

@pytest.fixture
def sample_data():
    # Crear datos de muestra para las pruebas
    data = pd.DataFrame({'Fecha': ['2023-01-01', '2023-01-02', '2023-01-03'],
                         'Precio': [10, 15, 20]})
    return data

def test_plot_moving_averages(sample_data):
    # Llamar a la función con los datos de muestra y verificar si se devuelve el DataFrame y la figura correctamente
    data, fig = plot_moving_averages(sample_data, 'Precio')
    assert isinstance(data, pd.DataFrame)
    assert isinstance(fig, go.Figure)

    # Verificar si se agregaron las columnas de las medias móviles correctamente
    assert 'MA_8' in data.columns
    assert 'MA_21' in data.columns
    assert 'MA_30' in data.columns
    assert 'MA_50' in data.columns
    assert 'MA_100' in data.columns
    assert 'MA_200' in data.columns

    # Verificar si la longitud de los datos de muestra es igual a la longitud del DataFrame devuelto
    assert len(data) == len(sample_data)

    # Verificar si se agregaron las líneas de las medias móviles en el gráfico
    assert len(fig.data) == 7  # El número de líneas en el gráfico debe ser igual a 7


def test_plot_moving_averages_custom_medias_moviles(sample_data):
    # Llamar a la función con datos de muestra y medias móviles personalizadas
    medias_moviles = [5, 10, 15]
    colores = ['red', 'green', 'blue']
    data, fig = plot_moving_averages(sample_data, 'Precio', medias_moviles=medias_moviles, colores=colores)

    # Verificar si se agregaron las columnas de las medias móviles personalizadas correctamente
    for ma in medias_moviles:
        assert f'MA_{ma}' in data.columns

    # Verificar si la longitud de los datos de muestra es igual a la longitud del DataFrame devuelto
    assert len(data) == len(sample_data)

    # Verificar si se agregaron las líneas de las medias móviles personalizadas en el gráfico
    assert len(fig.data) == len(medias_moviles) + 1  # El número de líneas en el gráfico debe ser igual al número de medias móviles personalizadas
