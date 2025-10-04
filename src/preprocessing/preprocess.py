import os
import sys

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa import seasonal

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (  # noqa: E402
    adf_test,
    crear_features_completas,
    crear_variable_exogena_segura,
    create_sequences,
    preparar_datos_prophet,
    verificar_no_data_leakage,
)


class TimeSeriesPreprocessor:
    """
    Comprehensive time series preprocessing class for multiple models
    """

    def __init__(self, config=None):
        """
        Initialize preprocessor with configuration

        Args:
            config (dict): Configuration dictionary for feature engineering
        """
        self.config = config or self._get_default_config()
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.is_fitted = False

    def _get_default_config(self):
        """Default configuration for feature engineering"""
        return {
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

    def load_data(self, file_path):
        """
        Load time series data from parquet file

        Args:
            file_path (str): Path to parquet file

        Returns:
            pd.DataFrame: Loaded data
        """
        data = pd.read_parquet(file_path)
        data = data.reset_index()
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.set_index("Date")
        return data

    def check_stationarity(self, series, title=""):
        """
        Check stationarity of time series

        Args:
            series (pd.Series): Time series to check
            title (str): Title for the test

        Returns:
            bool: True if stationary, False otherwise
        """
        return adf_test(series, title)

    def decompose_series(self, series, period=252, model="additive"):
        """
        Perform seasonal decomposition

        Args:
            series (pd.Series): Time series to decompose
            period (int): Seasonal period
            model (str): 'additive' or 'multiplicative'

        Returns:
            statsmodels decomposition object
        """
        return seasonal.seasonal_decompose(series, period=period, model=model)

    def create_features(self, series, target_col="Close"):
        """
        Create comprehensive features for machine learning models

        Args:
            series (pd.Series): Input time series
            target_col (str): Name of target column

        Returns:
            pd.DataFrame: DataFrame with features
        """
        # Create target series
        if isinstance(series, pd.Series):
            target_series = series
        else:
            target_series = series[target_col]

        # Create features
        features_df = crear_features_completas(target_series, self.config)

        # Add differenced target for some models
        features_df["target_diff"] = features_df["target"].diff()

        # Verify no data leakage
        verificar_no_data_leakage(features_df, "target", mostrar_ejemplo=False)

        return features_df

    def prepare_arima_data(self, series, start_date="2000-01-01"):
        """
        Prepare data for ARIMA models

        Args:
            series (pd.Series): Input time series
            start_date (str): Start date for filtering

        Returns:
            pd.Series: Prepared time series
        """
        if isinstance(series, pd.DataFrame):
            series = series["Close"]

        # Filter by date
        filtered_series = series[series.index >= start_date]

        # Check stationarity
        is_stationary = self.check_stationarity(
            filtered_series, "Original Series"
        )

        if not is_stationary:
            # Apply differencing
            diff_series = filtered_series.diff().dropna()
            self.check_stationarity(diff_series, "Differenced Series")
            return diff_series

        return filtered_series

    def prepare_prophet_data(self, series, events_periods=None):
        """
        Prepare data for Prophet model

        Args:
            series (pd.Series): Input time series
            events_periods (list): List of event periods for exogenous
                variables

        Returns:
            tuple: (prophet_df, exogenous_variables)
        """
        if isinstance(series, pd.DataFrame):
            series = series["Close"]

        # Create exogenous variables if events are provided
        exogenous_vars = {}
        if events_periods:
            for i, period in enumerate(events_periods):
                var_name = f"event_{i + 1}"
                exogenous_vars[var_name] = crear_variable_exogena_segura(
                    series, [period], var_name
                )

        # Prepare Prophet format
        prophet_df = preparar_datos_prophet(series, exogenous_vars)

        return prophet_df, list(exogenous_vars.keys())

    def prepare_ml_data(
        self,
        series,
        train_start="2021-01-01",
        train_end="2023-12-31",
        test_start="2024-01-01",
        target_col="Close",
    ):
        """
        Prepare data for machine learning models (CatBoost, LightGBM)

        Args:
            series (pd.Series or pd.DataFrame): Input data
            train_start (str): Training start date
            train_end (str): Training end date
            test_start (str): Test start date
            target_col (str): Target column name

        Returns:
            tuple: (train_features, train_target, test_features, test_target)
        """
        # Create features
        if isinstance(series, pd.DataFrame):
            target_series = series[target_col]
        else:
            target_series = series

        features_df = self.create_features(target_series, target_col)

        # Split data
        train_data = features_df.loc[train_start:train_end]
        test_data = features_df.loc[test_start:]

        # Remove rows with NaN values
        train_data = train_data.dropna()
        test_data = test_data.dropna()

        # Separate features and target
        feature_cols = [
            col
            for col in train_data.columns
            if col not in ["target", "target_diff"]
        ]

        train_features = train_data[feature_cols]
        train_target = train_data["target_diff"]  # Use differenced target

        test_features = test_data[feature_cols]
        test_target = test_data["target_diff"]

        return train_features, train_target, test_features, test_target

    def prepare_lstm_data(
        self,
        series,
        sequence_length=60,
        train_ratio=0.8,
        feature_columns=None,
        target_column="Close",
    ):
        """
        Prepare data for LSTM model

        Args:
            series (pd.DataFrame): Input data
            sequence_length (int): Length of input sequences
            train_ratio (float): Ratio for train/test split
            feature_columns (list): List of feature columns to use
            target_column (str): Target column name

        Returns:
            tuple: Prepared data for LSTM training
        """
        # Select features
        if feature_columns is None:
            feature_columns = [
                col for col in series.columns if col != target_column
            ]

        features = series[feature_columns].values
        target = series[target_column].values.reshape(-1, 1)

        # Scale features and target
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target)

        # Create sequences
        features_x, features_y = create_sequences(
            features_scaled, target_scaled.flatten(), sequence_length
        )

        # Split into train and test
        split_idx = int(len(features_x) * train_ratio)

        train_x = features_x[:split_idx]
        test_x = features_x[split_idx:]
        train_y = features_y[:split_idx]
        test_y = features_y[split_idx:]

        self.is_fitted = True

        return train_x, test_x, train_y, test_y

    def inverse_transform_lstm(self, predictions):
        """
        Inverse transform LSTM predictions

        Args:
            predictions (np.array): Scaled predictions

        Returns:
            np.array: Original scale predictions
        """
        if not self.is_fitted:
            raise ValueError(
                "Scaler not fitted. Call prepare_lstm_data first."
            )

        return self.target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()

    def create_sarimax_exogenous(self, series, events_periods):
        """
        Create exogenous variables for SARIMAX model

        Args:
            series (pd.Series): Input time series
            events_periods (list): List of event periods

        Returns:
            pd.DataFrame: Exogenous variables
        """
        exog_df = pd.DataFrame(index=series.index)

        for i, period in enumerate(events_periods):
            var_name = f"event_{i + 1}"
            exog_var = crear_variable_exogena_segura(
                series, [period], var_name
            )
            exog_df[var_name] = exog_var

        return exog_df


def preprocess_for_model(data_path, model_type, **kwargs):
    """
    Convenience function to preprocess data for specific model types

    Args:
        data_path (str): Path to data file
        model_type (str): Type of model ('arima', 'prophet', 'ml',
            'lstm', 'sarimax')
        **kwargs: Additional arguments for preprocessing

    Returns:
        Preprocessed data appropriate for the model type
    """
    preprocessor = TimeSeriesPreprocessor()
    data = preprocessor.load_data(data_path)

    if model_type == "arima":
        return preprocessor.prepare_arima_data(data, **kwargs)
    elif model_type == "prophet":
        return preprocessor.prepare_prophet_data(data, **kwargs)
    elif model_type == "ml":
        return preprocessor.prepare_ml_data(data, **kwargs)
    elif model_type == "lstm":
        return preprocessor.prepare_lstm_data(data, **kwargs)
    elif model_type == "sarimax":
        series = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
        exog = preprocessor.create_sarimax_exogenous(
            series, kwargs.get("events_periods", [])
        )
        return series, exog
    else:
        raise ValueError(f"Unknown model type: {model_type}")
