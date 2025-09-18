import yfinance as yf
import pandas as pd

def extract_stock_data(ticker, period):
    data = yf.Ticker(ticker)
    all_data = data.history(period = period)
    return all_data

