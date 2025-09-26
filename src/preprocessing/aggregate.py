import pandas as pd

def calculate_monthly_returns(df):
    """
    Calcula los retornos mensuales compuestos a partir de datos de precios diarios.

    Parámetros:
    df (pd.DataFrame): DataFrame con un índice de tipo DatetimeIndex y una
                       columna de 'Close' prices.

    Retorna:
    pd.DataFrame: DataFrame con los retornos mensuales.
    """
    # 1. Calcular el retorno diario como un porcentaje
    df['Daily_Return'] = df['Close'].pct_change()

    # 2. Calcular los retornos mensuales compuestos
    # Se suma 1 para obtener el factor de crecimiento diario
    # Se agrupa por mes ('M') y se multiplican los factores de crecimiento con .prod()
    # Finalmente, se resta 1 para obtener el retorno porcentual del mes
    monthly_returns = (1 + df['Daily_Return']).resample('M').prod() - 1

    # Convertir a DataFrame y renombrar la columna para mayor claridad
    monthly_returns_df = monthly_returns.to_frame(name='Monthly_Return')
    
    return monthly_returns_df


def calculate_monthly_returns_afp(df_raw):
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