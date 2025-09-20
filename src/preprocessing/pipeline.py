from extract import extract_stock_data
from aggregate import calculate_monthly_returns

def generate_raw_data(ticker,period):
    data = extract_stock_data(ticker=ticker, period=period)
    data.to_parquet("data/raw/raw_stock_data.parquet")
    monthly_data = calculate_monthly_returns(data)
    monthly_data.to_parquet("data/aggregated/agg_stock_data.parquet")
    return "success"

def generate_gold_data():
    from extract import extract_gold_data
    gold_data = extract_gold_data()
    gold_data.to_parquet("data/raw/raw_gold_data.parquet")
    return "success"

generate_raw_data("SPY","max")
generate_gold_data();
    