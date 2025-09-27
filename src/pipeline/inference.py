import pandas as pd
import numpy as np
import torch
import pickle
import joblib
import warnings
from prophet import Prophet
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import sys
import os
from typing import Dict, Any, Tuple, Optional, Union
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import LSTMModel, calculate_metrics
from preprocessing.preprocess import TimeSeriesPreprocessor

warnings.filterwarnings('ignore')


class TimeSeriesInference:
    """
    Comprehensive inference class for multiple time series models
    """

    def __init__(self, preprocessor=None):
        """
        Initialize inference engine with preprocessor

        Args:
            preprocessor: TimeSeriesPreprocessor instance
        """
        self.preprocessor = preprocessor or TimeSeriesPreprocessor()
        self.models = {}
        self.predictions = {}

    def load_model(self, model_type, model_path):
        """
        Load a trained model from file

        Args:
            model_type (str): Type of model ('arima', 'prophet', 'catboost', 'lightgbm', 'lstm')
            model_path (str): Path to the saved model

        Returns:
            Loaded model
        """
        print(f"Loading {model_type} model from {model_path}")

        if model_type in ['arima', 'sarimax', 'prophet']:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_type == 'catboost':
            model = CatBoostRegressor()
            model.load_model(model_path)
        elif model_type == 'lightgbm':
            model = joblib.load(model_path)
        elif model_type == 'lstm':
            checkpoint = torch.load(model_path, map_location='cpu')
            model_config = checkpoint['model_config']
            model = LSTMModel(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.models[model_type] = model
        print(f"{model_type} model loaded successfully")
        return model

    def predict_arima(self, model, steps=30, return_conf_int=True):
        """
        Make predictions using ARIMA model

        Args:
            model: Trained ARIMA model
            steps (int): Number of steps to forecast
            return_conf_int (bool): Whether to return confidence intervals

        Returns:
            tuple: (predictions, confidence_intervals) or just predictions
        """
        print(f"Making ARIMA predictions for {steps} steps...")

        forecast = model.forecast(steps=steps)

        if return_conf_int:
            conf_int = model.get_forecast(steps=steps).conf_int()
            return forecast, conf_int

        return forecast

    def predict_sarimax(self, model, exog_future, steps=30, return_conf_int=True):
        """
        Make predictions using SARIMAX model

        Args:
            model: Trained SARIMAX model
            exog_future (pd.DataFrame): Future exogenous variables
            steps (int): Number of steps to forecast
            return_conf_int (bool): Whether to return confidence intervals

        Returns:
            tuple: (predictions, confidence_intervals) or just predictions
        """
        print(f"Making SARIMAX predictions for {steps} steps...")

        forecast = model.forecast(steps=steps, exog=exog_future)

        if return_conf_int:
            conf_int = model.get_forecast(steps=steps, exog=exog_future).conf_int()
            return forecast, conf_int

        return forecast

    def predict_prophet(self, model, periods=30, freq='D', include_history=False):
        """
        Make predictions using Prophet model

        Args:
            model: Trained Prophet model
            periods (int): Number of periods to forecast
            freq (str): Frequency of predictions ('D', 'W', 'M', etc.)
            include_history (bool): Whether to include historical predictions

        Returns:
            pd.DataFrame: Prophet forecast dataframe
        """
        print(f"Making Prophet predictions for {periods} periods...")

        future = model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
        forecast = model.predict(future)

        return forecast

    def predict_ml_models(self, model, test_features, model_type='catboost'):
        """
        Make predictions using machine learning models (CatBoost, LightGBM)

        Args:
            model: Trained ML model
            test_features (pd.DataFrame): Test features
            model_type (str): Type of ML model

        Returns:
            np.array: Predictions
        """
        print(f"Making {model_type} predictions...")

        if model_type == 'lightgbm':
            predictions = model.predict(test_features.astype(float))
        else:
            predictions = model.predict(test_features)

        return predictions

    def predict_lstm(self, model, X_test, target_scaler=None):
        """
        Make predictions using LSTM model

        Args:
            model: Trained LSTM model
            X_test (np.array): Test sequences
            target_scaler: Scaler for inverse transformation

        Returns:
            np.array: Predictions (inverse transformed if scaler provided)
        """
        print("Making LSTM predictions...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        predictions = []

        with torch.no_grad():
            for i in range(len(X_test)):
                sequence = torch.FloatTensor(X_test[i:i+1]).to(device)
                pred = model(sequence)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions).reshape(-1)

        # Inverse transform if scaler provided
        if target_scaler is not None:
            predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        return predictions

    def predict_next_day_lstm(self, model, last_sequence, target_scaler=None, feature_scaler=None):
        """
        Predict next day value using LSTM

        Args:
            model: Trained LSTM model
            last_sequence (np.array): Last sequence for prediction
            target_scaler: Target scaler for inverse transformation
            feature_scaler: Feature scaler for scaling input

        Returns:
            float: Next day prediction
        """
        print("Predicting next day with LSTM...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # Scale the sequence if scaler provided
        if feature_scaler is not None:
            last_sequence = feature_scaler.transform(last_sequence)

        # Convert to tensor and add batch dimension
        sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(sequence_tensor)
            pred_value = prediction.cpu().numpy()[0, 0]

        # Inverse transform if scaler provided
        if target_scaler is not None:
            pred_value = target_scaler.inverse_transform([[pred_value]])[0, 0]

        return pred_value

    def multi_step_forecast_lstm(self, model, initial_sequence, steps=30, target_scaler=None):
        """
        Make multi-step forecast using LSTM

        Args:
            model: Trained LSTM model
            initial_sequence (np.array): Initial sequence
            steps (int): Number of steps to forecast
            target_scaler: Target scaler for inverse transformation

        Returns:
            np.array: Multi-step predictions
        """
        print(f"Making {steps}-step LSTM forecast...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        predictions = []
        current_sequence = initial_sequence.copy()

        with torch.no_grad():
            for _ in range(steps):
                # Predict next value
                sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
                pred = model(sequence_tensor)
                pred_value = pred.cpu().numpy()[0, 0]
                predictions.append(pred_value)

                # Update sequence for next prediction
                # Add prediction to the sequence and remove first element
                new_row = current_sequence[-1].copy()
                new_row[0] = pred_value  # Assuming target is first feature
                current_sequence = np.vstack([current_sequence[1:], new_row])

        predictions = np.array(predictions)

        # Inverse transform if scaler provided
        if target_scaler is not None:
            predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        return predictions

    def evaluate_predictions(self, y_true, y_pred, model_name="Model"):
        """
        Evaluate model predictions

        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            model_name (str): Name of the model

        Returns:
            dict: Evaluation metrics
        """
        print(f"Evaluating {model_name} predictions...")

        metrics = calculate_metrics(y_true, y_pred)
        metrics['model_name'] = model_name

        print(f"{model_name} Evaluation Metrics:")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")

        return metrics

    def predict_all_models(self, data_path, models_dir="models", test_start='2024-01-01',
                          forecast_days=30):
        """
        Make predictions using all available models

        Args:
            data_path (str): Path to the data file
            models_dir (str): Directory containing saved models
            test_start (str): Start date for test period
            forecast_days (int): Number of days to forecast

        Returns:
            dict: Predictions from all models
        """
        print("Making predictions with all models...")

        # Load data
        data = self.preprocessor.load_data(data_path)
        all_predictions = {}
        all_metrics = {}

        # 1. ARIMA Model
        arima_path = os.path.join(models_dir, "arima_model.pkl")
        if os.path.exists(arima_path):
            try:
                arima_model = self.load_model('arima', arima_path)
                arima_pred = self.predict_arima(arima_model, steps=forecast_days, return_conf_int=False)
                all_predictions['arima'] = arima_pred
                print("ARIMA predictions completed")
            except Exception as e:
                print(f"Error with ARIMA predictions: {e}")

        # 2. Prophet Model
        prophet_path = os.path.join(models_dir, "prophet_model.pkl")
        if os.path.exists(prophet_path):
            try:
                prophet_model = self.load_model('prophet', prophet_path)
                prophet_forecast = self.predict_prophet(prophet_model, periods=forecast_days)
                all_predictions['prophet'] = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                print("Prophet predictions completed")
            except Exception as e:
                print(f"Error with Prophet predictions: {e}")

        # 3. Machine Learning Models
        try:
            # Prepare ML data
            train_X, train_y, test_X, test_y = self.preprocessor.prepare_ml_data(
                data, test_start=test_start
            )

            # CatBoost
            catboost_path = os.path.join(models_dir, "catboost_model")
            if os.path.exists(catboost_path):
                try:
                    catboost_model = self.load_model('catboost', catboost_path)
                    catboost_pred = self.predict_ml_models(catboost_model, test_X, 'catboost')
                    all_predictions['catboost'] = catboost_pred

                    # Evaluate if test data available
                    if len(test_y) > 0:
                        metrics = self.evaluate_predictions(test_y, catboost_pred, "CatBoost")
                        all_metrics['catboost'] = metrics
                    print("CatBoost predictions completed")
                except Exception as e:
                    print(f"Error with CatBoost predictions: {e}")

            # LightGBM
            lightgbm_path = os.path.join(models_dir, "lightgbm_model.pkl")
            if os.path.exists(lightgbm_path):
                try:
                    lightgbm_model = self.load_model('lightgbm', lightgbm_path)
                    lightgbm_pred = self.predict_ml_models(lightgbm_model, test_X, 'lightgbm')
                    all_predictions['lightgbm'] = lightgbm_pred

                    # Evaluate if test data available
                    if len(test_y) > 0:
                        metrics = self.evaluate_predictions(test_y, lightgbm_pred, "LightGBM")
                        all_metrics['lightgbm'] = metrics
                    print("LightGBM predictions completed")
                except Exception as e:
                    print(f"Error with LightGBM predictions: {e}")

        except Exception as e:
            print(f"Error preparing ML data: {e}")

        # 4. LSTM Model
        lstm_path = os.path.join(models_dir, "lstm_model.pth")
        if os.path.exists(lstm_path):
            try:
                # Prepare LSTM data
                X_train, X_test, y_train, y_test = self.preprocessor.prepare_lstm_data(data)

                lstm_model = self.load_model('lstm', lstm_path)
                lstm_pred = self.predict_lstm(lstm_model, X_test, self.preprocessor.target_scaler)
                all_predictions['lstm'] = lstm_pred

                # Evaluate if test data available
                if len(y_test) > 0:
                    y_test_original = self.preprocessor.inverse_transform_lstm(y_test)
                    metrics = self.evaluate_predictions(y_test_original, lstm_pred, "LSTM")
                    all_metrics['lstm'] = metrics
                print("LSTM predictions completed")
            except Exception as e:
                print(f"Error with LSTM predictions: {e}")

        # Store results
        self.predictions = all_predictions
        self.evaluation_metrics = all_metrics

        return all_predictions

    def create_forecast_dates(self, last_date, periods, freq='D'):
        """
        Create future dates for forecasting

        Args:
            last_date (datetime): Last date in the data
            periods (int): Number of periods to forecast
            freq (str): Frequency ('D', 'W', 'M', etc.)

        Returns:
            pd.DatetimeIndex: Future dates
        """
        return pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq=freq)

    def save_predictions(self, predictions, save_path):
        """
        Save predictions to file

        Args:
            predictions (dict): Dictionary of predictions
            save_path (str): Path to save predictions
        """
        print(f"Saving predictions to {save_path}")

        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame()

        for model_name, pred in predictions.items():
            if isinstance(pred, pd.DataFrame):
                # Prophet predictions
                predictions_df[f'{model_name}_prediction'] = pred['yhat']
                if 'yhat_lower' in pred.columns:
                    predictions_df[f'{model_name}_lower'] = pred['yhat_lower']
                    predictions_df[f'{model_name}_upper'] = pred['yhat_upper']
            else:
                # Other model predictions
                predictions_df[f'{model_name}_prediction'] = pred

        predictions_df.to_csv(save_path, index=False)
        print("Predictions saved successfully")

    def get_ensemble_prediction(self, predictions, method='mean', weights=None):
        """
        Create ensemble predictions from multiple models

        Args:
            predictions (dict): Dictionary of predictions
            method (str): Ensemble method ('mean', 'median', 'weighted')
            weights (dict): Weights for weighted ensemble

        Returns:
            np.array: Ensemble predictions
        """
        print(f"Creating ensemble predictions using {method} method...")

        # Extract numeric predictions only
        numeric_predictions = {}
        for model_name, pred in predictions.items():
            if isinstance(pred, pd.DataFrame) and 'yhat' in pred.columns:
                numeric_predictions[model_name] = pred['yhat'].values
            elif isinstance(pred, (np.ndarray, list)):
                numeric_predictions[model_name] = np.array(pred)

        if not numeric_predictions:
            raise ValueError("No numeric predictions found for ensemble")

        # Align predictions to same length
        min_length = min(len(pred) for pred in numeric_predictions.values())
        aligned_predictions = {name: pred[:min_length] for name, pred in numeric_predictions.items()}

        # Create ensemble
        pred_matrix = np.column_stack(list(aligned_predictions.values()))

        if method == 'mean':
            ensemble_pred = np.mean(pred_matrix, axis=1)
        elif method == 'median':
            ensemble_pred = np.median(pred_matrix, axis=1)
        elif method == 'weighted' and weights:
            weight_array = np.array([weights.get(name, 1.0) for name in aligned_predictions.keys()])
            weight_array = weight_array / np.sum(weight_array)  # Normalize
            ensemble_pred = np.average(pred_matrix, axis=1, weights=weight_array)
        else:
            ensemble_pred = np.mean(pred_matrix, axis=1)

        return ensemble_pred


if __name__ == "__main__":
    # Example usage
    inference = TimeSeriesInference()

    # Make predictions with all models
    data_path = "../preprocessing/data/raw/raw_stock_data.parquet"
    models_dir = "../../models"

    if os.path.exists(data_path):
        predictions = inference.predict_all_models(
            data_path,
            models_dir=models_dir,
            test_start='2024-01-01',
            forecast_days=30
        )

        print("\n=== Predictions Summary ===")
        for model_name, pred in predictions.items():
            if isinstance(pred, pd.DataFrame):
                print(f"{model_name}: {len(pred)} predictions (DataFrame)")
            else:
                print(f"{model_name}: {len(pred)} predictions")

        # Save predictions
        inference.save_predictions(predictions, "../../predictions/all_model_predictions.csv")

        # Create ensemble prediction
        if len(predictions) > 1:
            ensemble_pred = inference.get_ensemble_prediction(predictions)
            print(f"\nEnsemble prediction created with {len(ensemble_pred)} values")

    else:
        print(f"Data file not found: {data_path}")