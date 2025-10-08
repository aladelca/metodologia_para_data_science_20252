import pandas as pd


def calculate_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los retornos mensuales compuestos a partir
    de datos de precios diarios.

    Parámetros:
    df (pd.DataFrame): DataFrame con un índice de tipo DatetimeIndex y una
                       columna de 'Close' prices.

    Retorna:
    pd.DataFrame: DataFrame con los retornos mensuales.
    """
    # 1. Calcular el retorno diario como un porcentaje
    df["Daily_Return"] = df["Close"].pct_change()

    # 2. Calcular los retornos mensuales compuestos
    # Se suma 1 para obtener el factor de crecimiento diario
    # Se agrupa por mes ('M') y se multiplican los factores
    # de crecimiento con .prod()
    # Finalmente, se resta 1 para obtener el retorno porcentual del mes
    monthly_returns = (1 + df["Daily_Return"]).resample("M").prod() - 1
    # Convertir a DataFrame y renombrar la columna para mayor claridad
    monthly_returns_df = monthly_returns.to_frame(name="Monthly_Return")
    return monthly_returns_df