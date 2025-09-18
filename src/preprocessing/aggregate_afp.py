import pandas as pd

def calculate_monthly_returns(df_raw):
    """
    Calcula los retornos mensuales compuestos a partir de rentabilidades anuales.

    Parameters:
    annual_returns_df (pd.DataFrame): DataFrame con rentabilidades anuales de AFP

    Returns:
    pd.DataFrame: DataFrame con los retornos mensuales compuestos
    """
    # Calcular rentabilidad mensual: (1 + anual)^(1/12) - 1
    monthly_returns_df = (1 + df_raw) ** (1/12) - 1
    
    return monthly_returns_df