from extract_afp import extract_all_afp_data
from aggregate_afp import calculate_monthly_returns

def generate_raw_data():
    data = extract_all_afp_data()
    data.to_parquet("data/raw/raw_pipe_afp_data.parquet")
    monthly_data = calculate_monthly_returns(data)
    monthly_data.to_parquet("data/aggregated/agg_pipe_afp_data.parquet")
    return "success"

generate_raw_data()
    