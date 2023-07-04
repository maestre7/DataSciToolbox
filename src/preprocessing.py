import re
import pandas as pd

def split_and_encode_strings(column:pd.Series, use_encoding: bool = False ) -> pd.DataFrame:
    """
    Separa una columna de un DataFrame utilizando cualquier carácter que no sea una letra o un número como separador,
    y opcionalmente aplica one-hot encoding (get dummies) a las palabras separadas.

    Args:
        column (pd.Series): La columna del DataFrame que se desea separar y encodear.
        use_encoding (bool, optional): Indica si se debe aplicar one-hot encoding a las palabras separadas.
            Por defecto es False.

    Returns:
        pd.DataFrame: Un DataFrame con las palabras separadas en una sola columna si use_encoding es False,
            o un DataFrame con columnas correspondientes a cada palabra si use_encoding es True.

    """

    try:
        separador = re.compile(r'[^a-zA-Z0-9]+')
        palabras = column.apply(lambda x: separador.split(x))
        if use_encoding:
            df = pd.get_dummies(palabras.apply(pd.Series).stack()).groupby(level=1).sum()
        else:
            df = pd.DataFrame(palabras)
        return df
    except Exception as e:
        print("Ocurrió un error al separar y encodear las strings:", str(e))