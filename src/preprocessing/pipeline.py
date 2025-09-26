from extract import extract_stock_data, extract_all_afp_data
from aggregate import calculate_monthly_returns, calculate_monthly_returns_afp

def generate_raw_data(ticker,period):
    data = extract_stock_data(ticker=ticker, period=period)
    data.to_parquet("data/raw/raw_stock_data.parquet")
    monthly_data = calculate_monthly_returns(data)
    monthly_data.to_parquet("data/aggregated/agg_stock_data.parquet")
    return "success"

def generate_raw_data_afp():
    data = extract_all_afp_data()
    data.to_parquet("data/raw/raw_pipe_afp_data.parquet")
    monthly_data = calculate_monthly_returns_afp(data)
    monthly_data.to_parquet("data/aggregated/agg_pipe_afp_data.parquet")
    return "success"

generate_raw_data("SPY","max")
generate_raw_data_afp()
    