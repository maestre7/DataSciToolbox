import pandas as pd
import pytest
from matplotlib import pyplot as plt

from src.visualization import dist_variables

def test_dist_variables_target():
    # DataFrame de ejemplo
    data = pd.DataFrame({
        'Variable1': [1, 2, 3, 4, 5],
        'Variable2': [6, 7, 8, 9, 10],
        'Target': ['A', 'B', 'A', 'B', 'A']
    })

    fig = dist_variables(data, target='Target')

    assert isinstance(fig, plt.Figure) # type: ignore

# Prueba sin el uso de target
def test_dist_variables():
    # DataFrame de ejemplo
    data = pd.DataFrame({
        'Variable1': [1, 2, 3, 4, 5],
        'Variable2': [6, 7, 8, 9, 10]
    })

    fig = dist_variables(data)

    assert isinstance(fig, plt.Figure) # type: ignore
