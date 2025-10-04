import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.aggregate import calculate_monthly_returns  # noqa: E402


def test_calculate_monthly_returns():
    # Crear mock data con precios diarios para 3 meses
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    mock_data = pd.DataFrame(
        {"Close": [100, 101, 102, 105, 104, 106, 108, 107, 110, 112] * 9},
        index=dates,
    )

    result = calculate_monthly_returns(mock_data)

    assert isinstance(result, pd.DataFrame)
    assert "Monthly_Return" in result.columns
    assert len(result) == 3  # 3 meses
    assert result.index.freq is not None or len(result.index) > 0


def test_calculate_monthly_returns_single_month():
    # Mock data para un solo mes
    dates = pd.date_range("2023-01-01", periods=31, freq="D")
    mock_data = pd.DataFrame(
        {"Close": [100 + i for i in range(31)]},
        index=dates,  # Precios incrementales
    )

    result = calculate_monthly_returns(mock_data)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "Monthly_Return" in result.columns
    # Debería ser positivo por precios crecientes
    assert result.iloc[0]["Monthly_Return"] > 0


def test_calculate_monthly_returns_declining_prices():
    # Mock data con precios declinantes
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    mock_data = pd.DataFrame(
        {
            # Precios decrecientes
            "Close": [100 - i for i in range(30)]
        },
        index=dates,
    )

    result = calculate_monthly_returns(mock_data)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    # Debería ser negativo por precios decrecientes
    assert result.iloc[0]["Monthly_Return"] < 0
