import pandas as pd
import pytest

from src.preprocessing import create_dataframe_yahoo_finance

def test_create_dataframe_yahoo_finance():
    # Símbolo válido
    symbol = "AAPL"
    df = create_dataframe_yahoo_finance(symbol)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Símbolo inválido
    symbol_invalid = "INVALID"
    try:
        df_invalid = create_dataframe_yahoo_finance(symbol_invalid)
        assert df_invalid.empty
    except Exception as e:
        pytest.fail(f"Se esperaba una excepción, pero se capturó: {str(e)}")
