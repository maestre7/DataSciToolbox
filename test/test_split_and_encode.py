from src.preprocessing import split_and_encode_strings
import pandas as pd
import pytest


datos = {
    'Col1': ['perro,gato,conejo', 'manzana,naranja', 'sol,luna,estrella'],
    'Col2': ['rojo,verde', 'azul', 'amarillo,rosa,morado'],
    'Col3': ['458,coche,moto', 'ciudad,pueblo', 'playa,montaña']
}

# Crear el DataFrame
df = pd.DataFrame(datos)


def test_split_and_encode_1():

    # Test case 1
    split_and_encode_strings(df['Col1'],True)
    
def test_split_and_encode_2():

    # Test case 2
    split_and_encode_strings(df['Col2'],True)

def test_split_and_encode_3():

    # Test case 3
    split_and_encode_strings(df['Col3'],True)


