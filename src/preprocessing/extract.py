import yfinance as yf
import pandas as pd
from typing import Any

def extract_stock_data(ticker : str, period : str) -> Any:
    data = yf.Ticker(ticker)
    all_data = data.history(period = period)
    return all_data

def extract_gold_data() -> Any:
    """
    Extraer historial del costo del oro desde https://datahub.io/core/gold-prices

    Returns:
        pd.DataFrame: Un DataFrame que contiene los precios históricos del oro con columnas 'Date' y 'Price'.
    """

    url = "https://datahub.io/core/gold-prices/r/monthly.csv"
    gold_data = pd.read_csv(url)
    gold_data['Date'] = pd.to_datetime(gold_data['Date'])
    return gold_data
