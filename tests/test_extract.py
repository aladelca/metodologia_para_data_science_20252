import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.extract import (  # noqa: E402
    extract_gold_data,
    extract_stock_data,
)


def test_extract_stock_data_aapl():
    result = extract_stock_data("AAPL", "1y")

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "Close" in result.columns
    assert "Open" in result.columns
    assert "High" in result.columns
    assert "Low" in result.columns
    assert "Volume" in result.columns


def test_extract_gold_data_aapl():
    result = extract_gold_data()

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "Date" in result.columns
