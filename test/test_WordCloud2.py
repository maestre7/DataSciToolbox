import pytest
from src.visualization import plot_wordcloud

def test_plot_wordcloud():
    # Caso de prueba con datos válidos
    text = 'hola hola hola adiós adiós'
    figsize = (8, 6)
    filename = 'test.png'
    plot_wordcloud(text, figsize, filename)

    # Caso de prueba con datos inválidos
    with pytest.raises(Exception):
        text = 1234
        figsize = (8, 6)
        filename = 'test.png'
        plot_wordcloud(text, figsize, filename)