import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def adf_test(timeseries, title=""):
    """
    Augmented Dickey-Fuller test for stationarity
    """
    print(f"Results of Augmented Dickey-Fuller Test: {title}")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput[f"Critical Value ({key})"] = value

    print(dfoutput)
    if dftest[1] <= 0.05:
        print("Series is stationary")
    else:
        print("Series is not stationary")
    return dftest[1] <= 0.05


def crear_variables_lag(serie, lags=None, prefijo="lag"):
    """Creates lag variables avoiding data leakage"""
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20]

    df = pd.DataFrame(index=serie.index)
    for lag in lags:
        df[f"{prefijo}_{lag}"] = serie.shift(lag)

    return df


def crear_promedios_moviles(serie, ventanas=None, prefijo="ma"):
    """Creates moving averages avoiding data leakage"""
    if ventanas is None:
        ventanas = [5, 10, 20, 50, 100]

    df = pd.DataFrame(index=serie.index)
    for ventana in ventanas:
        df[f"{prefijo}_{ventana}"] = (
            serie.shift(1).rolling(window=ventana, min_periods=1).mean()
        )

    return df


def crear_std_moviles(serie, ventanas=None, prefijo="std"):
    """Creates moving standard deviations"""
    if ventanas is None:
        ventanas = [5, 10, 20, 50]

    df = pd.DataFrame(index=serie.index)
    for ventana in ventanas:
        df[f"{prefijo}_{ventana}"] = (
            serie.shift(1).rolling(window=ventana, min_periods=1).std()
        )

    return df


def crear_variables_rolling_avanzadas(serie, ventanas=None, prefijo="roll"):
    """Creates advanced rolling statistics"""
    if ventanas is None:
        ventanas = [5, 10, 20]

    df = pd.DataFrame(index=serie.index)
    for ventana in ventanas:
        rolling = serie.shift(1).rolling(window=ventana, min_periods=1)
        df[f"{prefijo}_min_{ventana}"] = rolling.min()
        df[f"{prefijo}_max_{ventana}"] = rolling.max()
        df[f"{prefijo}_skew_{ventana}"] = rolling.skew()
        df[f"{prefijo}_kurt_{ventana}"] = rolling.kurt()

    return df


def crear_variables_tecnicas(
    serie, ventanas_cortas=None, ventanas_largas=None
):
    """Creates technical indicators"""
    if ventanas_cortas is None:
        ventanas_cortas = [5, 10]
    if ventanas_largas is None:
        ventanas_largas = [20, 50]

    df = pd.DataFrame(index=serie.index)

    for v_corta in ventanas_cortas:
        for v_larga in ventanas_largas:
            if v_corta < v_larga:
                ma_corta = serie.shift(1).rolling(window=v_corta).mean()
                ma_larga = serie.shift(1).rolling(window=v_larga).mean()
                df[f"ma_diff_{v_corta}_{v_larga}"] = ma_corta - ma_larga
                df[f"ma_ratio_{v_corta}_{v_larga}"] = ma_corta / ma_larga

    return df


def crear_variables_fechas_basicas(serie):
    """Creates basic date features"""
    df = pd.DataFrame(index=serie.index)
    fecha = serie.index

    df["year"] = fecha.year
    df["month"] = fecha.month
    df["day"] = fecha.day
    df["dayofweek"] = fecha.dayofweek
    df["quarter"] = fecha.quarter
    df["is_weekend"] = (fecha.dayofweek >= 5).astype(int)
    df["is_month_start"] = fecha.is_month_start.astype(int)
    df["is_month_end"] = fecha.is_month_end.astype(int)
    df["is_quarter_start"] = fecha.is_quarter_start.astype(int)
    df["is_quarter_end"] = fecha.is_quarter_end.astype(int)
    df["is_year_start"] = fecha.is_year_start.astype(int)
    df["is_year_end"] = fecha.is_year_end.astype(int)

    return df


def crear_variables_fechas_especiales(serie):
    """Creates special date features"""
    df = pd.DataFrame(index=serie.index)
    fecha = serie.index

    # Días especiales del mercado financiero
    df["es_lunes"] = (fecha.dayofweek == 0).astype(int)
    df["es_viernes"] = (fecha.dayofweek == 4).astype(int)
    df["dias_desde_lunes"] = fecha.dayofweek
    df["dias_hasta_viernes"] = 4 - fecha.dayofweek

    return df


def crear_variables_estacionales(serie):
    """Creates seasonal features"""
    df = pd.DataFrame(index=serie.index)
    fecha = serie.index

    df["sin_month"] = np.sin(2 * np.pi * fecha.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * fecha.month / 12)
    df["sin_day"] = np.sin(2 * np.pi * fecha.day / 31)
    df["cos_day"] = np.cos(2 * np.pi * fecha.day / 31)
    df["sin_dayofweek"] = np.sin(2 * np.pi * fecha.dayofweek / 7)
    df["cos_dayofweek"] = np.cos(2 * np.pi * fecha.dayofweek / 7)

    return df


def crear_variables_mercado_financiero(serie):
    """Creates financial market specific features"""
    df = pd.DataFrame(index=serie.index)
    fecha = serie.index

    # Efectos de calendario financiero
    df["es_inicio_mes"] = fecha.is_month_start.astype(int)
    df["es_fin_mes"] = fecha.is_month_end.astype(int)
    df["es_inicio_trimestre"] = fecha.is_quarter_start.astype(int)
    df["es_fin_trimestre"] = fecha.is_quarter_end.astype(int)
    df["es_inicio_año"] = fecha.is_year_start.astype(int)
    df["es_fin_año"] = fecha.is_year_end.astype(int)

    # Semana del mes
    df["semana_del_mes"] = ((fecha.day - 1) // 7) + 1

    return df


def calcular_momentum_simple(serie, ventanas=None, prefijo="momentum"):
    """Calculates simple momentum indicators"""
    if ventanas is None:
        ventanas = [5, 10, 20]

    df = pd.DataFrame(index=serie.index)
    for ventana in ventanas:
        df[f"{prefijo}_{ventana}"] = serie / serie.shift(ventana) - 1

    return df


def calcular_velocidad_momentum_corregida(
    serie, ventanas=None, metodo="lineal", columna_target="target"
):
    """Calculates momentum velocity"""
    if ventanas is None:
        ventanas = [5, 10, 20]

    df = pd.DataFrame(index=serie.index)

    for ventana in ventanas:
        if metodo == "lineal":
            df[f"velocidad_{ventana}"] = serie.diff(ventana) / ventana
        elif metodo == "log":
            df[f"velocidad_log_{ventana}"] = (
                np.log(serie) - np.log(serie.shift(ventana))
            ) / ventana
        elif metodo == "pct":
            df[f"velocidad_pct_{ventana}"] = (
                serie.pct_change(ventana) / ventana
            )

    return df


def crear_features_completas(serie, config=None):
    """Main function that creates all features avoiding data leakage"""
    if config is None:
        config = {
            "lags": [1, 2, 3, 5, 10, 20],
            "ventanas_ma": [5, 10, 20, 50, 100],
            "ventanas_std": [5, 10, 20, 50],
            "ventanas_roll": [5, 10, 20],
            "ventanas_cortas": [5, 10],
            "ventanas_largas": [20, 50],
            "ventanas_momentum": [5, 10, 20],
            "incluir_fechas": True,
            "incluir_tecnicas": True,
            "incluir_momentum": True,
        }

    # Target variable
    df_final = pd.DataFrame(index=serie.index)
    df_final["target"] = serie

    # Lag variables
    if "lags" in config:
        df_lags = crear_variables_lag(serie, config["lags"])
        df_final = pd.concat([df_final, df_lags], axis=1)

    # Moving averages
    if "ventanas_ma" in config:
        df_ma = crear_promedios_moviles(serie, config["ventanas_ma"])
        df_final = pd.concat([df_final, df_ma], axis=1)

    # Standard deviations
    if "ventanas_std" in config:
        df_std = crear_std_moviles(serie, config["ventanas_std"])
        df_final = pd.concat([df_final, df_std], axis=1)

    # Advanced rolling
    if "ventanas_roll" in config:
        df_roll = crear_variables_rolling_avanzadas(
            serie, config["ventanas_roll"]
        )
        df_final = pd.concat([df_final, df_roll], axis=1)

    # Technical indicators
    if config.get("incluir_tecnicas", True):
        df_tech = crear_variables_tecnicas(
            serie, config.get("ventanas_cortas"), config.get("ventanas_largas")
        )
        df_final = pd.concat([df_final, df_tech], axis=1)

    # Date features
    if config.get("incluir_fechas", True):
        df_fechas_basicas = crear_variables_fechas_basicas(serie)
        df_fechas_especiales = crear_variables_fechas_especiales(serie)
        df_fechas_estacionales = crear_variables_estacionales(serie)
        df_fechas_mercado = crear_variables_mercado_financiero(serie)

        df_final = pd.concat(
            [
                df_final,
                df_fechas_basicas,
                df_fechas_especiales,
                df_fechas_estacionales,
                df_fechas_mercado,
            ],
            axis=1,
        )

    # Momentum indicators
    if config.get("incluir_momentum", True):
        df_momentum = calcular_momentum_simple(
            serie, config.get("ventanas_momentum")
        )
        df_velocidad = calcular_velocidad_momentum_corregida(
            serie, config.get("ventanas_momentum")
        )
        df_final = pd.concat([df_final, df_momentum, df_velocidad], axis=1)

    return df_final


def verificar_no_data_leakage(
    features, target_col="target", mostrar_ejemplo=True
):
    """Verifies no data leakage in created features"""
    print("Verificando que no hay data leakage...")

    # Check correlation with future values
    correlaciones_futuras = []
    for col in features.columns:
        if col != target_col:
            future_corr = features[col].corr(features[target_col].shift(-1))
            if not pd.isna(future_corr):
                correlaciones_futuras.append((col, future_corr))

    correlaciones_futuras.sort(
        key=lambda param_x: abs(param_x[1]), reverse=True
    )

    if mostrar_ejemplo and correlaciones_futuras:
        print("Top 5 correlaciones con valores futuros:")
        for i, (col, corr) in enumerate(correlaciones_futuras[:5]):
            print(f"{i + 1}. {col}: {corr: .4f}")

    return correlaciones_futuras


def crear_variable_exogena_segura(
    serie, periodos_eventos, nombre_evento="evento"
):
    """Creates exogenous variables for important events"""
    variable_exogena = pd.Series(0, index=serie.index, name=nombre_evento)

    for periodo in periodos_eventos:
        if isinstance(periodo, str):
            variable_exogena.loc[periodo] = 1
        elif isinstance(periodo, (list, tuple)) and len(periodo) == 2:
            variable_exogena.loc[periodo[0] : periodo[1]] = 1

    return variable_exogena


def auto_arima_manual(serie, max_p=30, max_d=2, max_q=30, criterio="aic"):
    """Manual implementation of auto_arima using grid search"""
    best_aic = np.inf
    best_bic = np.inf
    best_order = None
    best_model = None

    results = []

    for param_p in range(max_p + 1):
        for param_d in range(max_d + 1):
            for param_q in range(max_q + 1):
                try:
                    model = ARIMA(serie, order=(param_p, param_d, param_q))
                    fitted_model = model.fit()

                    aic = fitted_model.aic
                    bic = fitted_model.bic

                    results.append(
                        {
                            "order": (param_p, param_d, param_q),
                            "aic": aic,
                            "bic": bic,
                        }
                    )

                    if criterio == "aic" and aic < best_aic:
                        best_aic = aic
                        best_order = (param_p, param_d, param_q)
                        best_model = fitted_model
                    elif criterio == "bic" and bic < best_bic:
                        best_bic = bic
                        best_order = (param_p, param_d, param_q)
                        best_model = fitted_model

                except Exception:  # nosec B112
                    continue

    print(f"Best order: {best_order}")
    best_value = best_aic if criterio == "aic" else best_bic
    print(f"Best {criterio.upper()}: {best_value}")

    return best_model, best_order, pd.DataFrame(results)


def preparar_datos_prophet(serie, variables_exogenas=None):
    """Prepares data in Prophet format (ds, y columns)"""
    df = pd.DataFrame({"ds": serie.index, "y": serie.values})

    if variables_exogenas is not None:
        for var_name, var_series in variables_exogenas.items():
            df[var_name] = var_series.values

    return df


class LSTMModel(nn.Module):  # type: ignore
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
    ) -> None:
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(
            self.num_layers, input_data.size(0), self.hidden_size
        ).to(input_data.device)
        c0 = torch.zeros(
            self.num_layers, input_data.size(0), self.hidden_size
        ).to(input_data.device)

        out, _ = self.lstm(input_data, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def create_sequences(features, target, sequence_length):
    """Creates sequences for LSTM training"""
    sequences_features, sequences_target = [], []
    for i in range(sequence_length, len(features)):
        sequences_features.append(features[i - sequence_length : i])
        sequences_target.append(target[i])
    return np.array(sequences_features), np.array(sequences_target)


def calculate_metrics(true_values, predicted_values):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mape = (
        np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    )

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


def handle_timezone_compatibility(date_string, data_index):
    """
    Handle timezone compatibility between date strings and data index

    Args:
        date_string (str): Date string to convert
        data_index (pd.DatetimeIndex): Data index to match timezone with

    Returns:
        pd.Timestamp: Timezone-compatible timestamp
    """
    date_dt = pd.to_datetime(date_string)

    # Handle timezone-aware data index
    if data_index.tz is not None:
        # If data index is timezone-aware, make the date timezone-aware too
        if date_dt.tz is None:
            date_dt = date_dt.tz_localize(data_index.tz)
        else:
            # Convert to the same timezone as data index
            date_dt = date_dt.tz_convert(data_index.tz)
    else:
        # If data index is timezone-naive, ensure date is also timezone-naive
        if date_dt.tz is not None:
            date_dt = date_dt.tz_localize(None)

    return date_dt
