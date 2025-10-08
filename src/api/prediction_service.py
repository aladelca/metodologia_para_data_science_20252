"""
Prediction service for handling model inference
"""

import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pipeline.inference import TimeSeriesInference  # noqa: E402
from preprocessing.preprocess import TimeSeriesPreprocessor  # noqa: E402

from .prediction_models import (  # noqa: E402
    AvailableModelsResponse,
    ModelPrediction,
    PredictionRequest,
    PredictionResponse,
)


class PredictionService:
    """Service for handling prediction requests"""

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize prediction service

        Args:
            models_dir: Directory containing trained models
        """
        # Set models directory relative to project root
        if models_dir is None:
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            self.models_dir = os.path.join(
                project_root, "src", "models", "training"
            )
        else:
            self.models_dir = models_dir

        self.preprocessor = TimeSeriesPreprocessor()
        self.inference = TimeSeriesInference(self.preprocessor)
        self.default_data_path = os.path.join(
            Path(__file__).parent.parent,
            "preprocessing",
            "data",
            "raw",
            "raw_stock_data.parquet",
        )

    def get_available_models(self) -> AvailableModelsResponse:
        """
        Get list of available trained models

        Returns:
            AvailableModelsResponse with model information
        """
        if not os.path.exists(self.models_dir):
            return AvailableModelsResponse(
                models_directory=self.models_dir,
                available_models=[],
                model_details=[],
                total_models=0,
            )

        # Search for different model file types
        model_patterns = {
            "arima": "arima_model.pkl",
            "sarimax": "sarimax_model.pkl",
            "prophet": "prophet_model.pkl",
            "catboost": "catboost_model",
            "lightgbm": "lightgbm_model.pkl",
            "lstm": "lstm_model.pth",
        }

        available_models = []
        model_details = []

        for model_type, pattern in model_patterns.items():
            model_path = os.path.join(self.models_dir, pattern)
            if os.path.exists(model_path):
                file_stats = os.stat(model_path)
                available_models.append(pattern)
                model_details.append(
                    {
                        "model_name": pattern,
                        "model_type": model_type,
                        "file_path": model_path,
                        "file_size_mb": round(
                            file_stats.st_size / (1024 * 1024), 2
                        ),
                        "last_modified": datetime.fromtimestamp(
                            file_stats.st_mtime
                        ).isoformat(),
                    }
                )

        return AvailableModelsResponse(
            models_directory=self.models_dir,
            available_models=available_models,
            model_details=model_details,
            total_models=len(available_models),
        )

    def _determine_model_type(self, model_name: str) -> str:
        """Determine model type from filename"""
        model_name_lower = model_name.lower()
        if "arima" in model_name_lower:
            return "arima"
        elif "sarimax" in model_name_lower:
            return "sarimax"
        elif "prophet" in model_name_lower:
            return "prophet"
        elif "catboost" in model_name_lower:
            return "catboost"
        elif "lightgbm" in model_name_lower:
            return "lightgbm"
        elif "lstm" in model_name_lower:
            return "lstm"
        else:
            return "unknown"

    def _get_forecast_dates(
        self, start_date: datetime, periods: int
    ) -> List[str]:
        """Generate forecast dates"""
        dates = pd.date_range(start=start_date, periods=periods, freq="D")
        return [date.strftime("%Y-%m-%d") for date in dates]

    async def predict_single_model(
        self, model_name: str, request: PredictionRequest
    ) -> ModelPrediction:
        """
        Make predictions using a single model

        Args:
            model_name: Name of the model file
            request: Prediction request

        Returns:
            ModelPrediction with results
        """
        start_time = time.time()

        # Build full path
        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Determine model type
        model_type = self._determine_model_type(model_name)

        # Load data
        data_path = request.data_path or self.default_data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data = self.preprocessor.load_data(data_path)
        last_date = data.index[-1]
        forecast_dates = self._get_forecast_dates(
            last_date + timedelta(days=1), request.forecast_days
        )

        # Load model
        model = self.inference.load_model(model_type, model_path)  # type: ignore[no-untyped-call]

        # Make predictions based on model type
        predictions = []
        confidence_lower = None
        confidence_upper = None

        if model_type in ["arima", "sarimax"]:
            if request.include_confidence_intervals:
                pred, conf_int = self.inference.predict_arima(  # type: ignore[no-untyped-call]
                    model, steps=request.forecast_days, return_conf_int=True
                )
                predictions = pred.tolist()
                confidence_lower = conf_int.iloc[:, 0].tolist()
                confidence_upper = conf_int.iloc[:, 1].tolist()
            else:
                pred = self.inference.predict_arima(  # type: ignore[no-untyped-call]
                    model, steps=request.forecast_days, return_conf_int=False
                )
                predictions = pred.tolist()

        elif model_type == "prophet":
            forecast = self.inference.predict_prophet(  # type: ignore[no-untyped-call]
                model, periods=request.forecast_days, freq="D"
            )
            # Get only future predictions (not historical)
            future_forecast = forecast.tail(request.forecast_days)
            predictions = future_forecast["yhat"].tolist()

            if request.include_confidence_intervals:
                confidence_lower = future_forecast["yhat_lower"].tolist()
                confidence_upper = future_forecast["yhat_upper"].tolist()

        elif model_type in ["catboost", "lightgbm"]:
            # For ML models, we need to prepare features for future dates
            # This is a simplified approach - in production you'd need
            # proper feature engineering for future dates
            _, _, test_X, _ = self.preprocessor.prepare_ml_data(  # type: ignore[no-untyped-call]
                data, test_start=last_date.strftime("%Y-%m-%d")
            )

            if len(test_X) > 0:
                # Use available test data
                num_predictions = min(len(test_X), request.forecast_days)
                pred = self.inference.predict_ml_models(  # type: ignore[no-untyped-call]
                    model, test_X[:num_predictions], model_type
                )
                predictions = pred.tolist()
                forecast_dates = forecast_dates[:num_predictions]
            else:
                raise ValueError(
                    f"No test data available for {model_type} predictions"
                )

        elif model_type == "lstm":
            # Prepare LSTM data
            _, X_test, _, y_test = self.preprocessor.prepare_lstm_data(data)  # type: ignore[no-untyped-call]

            if len(X_test) > 0:
                # Multi-step forecast
                pred = self.inference.multi_step_forecast_lstm(  # type: ignore[no-untyped-call]
                    model,
                    X_test[-1],
                    steps=request.forecast_days,
                    target_scaler=self.preprocessor.target_scaler,
                )
                predictions = pred.tolist()
            else:
                raise ValueError("No data available for LSTM predictions")

        prediction_time = time.time() - start_time

        return ModelPrediction(
            model_name=model_name,
            model_type=model_type,
            predictions=predictions,
            dates=forecast_dates[: len(predictions)],
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            prediction_time_seconds=round(prediction_time, 3),
        )

    async def predict(
        self, request: PredictionRequest
    ) -> PredictionResponse:
        """
        Make predictions based on request

        Args:
            request: Prediction request

        Returns:
            PredictionResponse with predictions
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Get available models
        available_models_response = self.get_available_models()

        if available_models_response.total_models == 0:
            raise ValueError(
                f"No trained models found in {self.models_dir}"
            )

        # Determine which models to use
        if request.model_name:
            # Use specific model
            available_models = available_models_response.available_models
            if request.model_name not in available_models:
                raise ValueError(
                    f"Model '{request.model_name}' not found. "
                    f"Available models: {available_models}"
                )
            models_to_use = [request.model_name]
        else:
            # Use all available models
            models_to_use = available_models_response.available_models

        # Make predictions with each model
        predictions = []
        successful = 0
        failed = 0

        for model_name in models_to_use:
            try:
                pred = await self.predict_single_model(model_name, request)
                predictions.append(pred)
                successful += 1
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                failed += 1

        if len(predictions) == 0:
            raise ValueError("All model predictions failed")

        # Calculate ensemble prediction if multiple models
        ensemble_prediction = None
        ensemble_dates = None

        if len(predictions) > 1:
            # Create predictions dictionary for ensemble
            preds_dict = {
                pred.model_name: np.array(pred.predictions)
                for pred in predictions
            }

            ensemble_pred = self.inference.get_ensemble_prediction(  # type: ignore[no-untyped-call]
                preds_dict, method=request.ensemble_method or "mean"
            )
            ensemble_prediction = ensemble_pred.tolist()

            # Use dates from first prediction
            ensemble_dates = predictions[0].dates[: len(ensemble_prediction)]

        # Determine forecast date range
        data_path = request.data_path or self.default_data_path
        data = self.preprocessor.load_data(data_path)  # type: ignore[no-untyped-call]
        last_date = data.index[-1]
        forecast_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        forecast_end = (
            last_date + timedelta(days=request.forecast_days)
        ).strftime("%Y-%m-%d")

        total_time = time.time() - start_time

        return PredictionResponse(
            request_id=request_id,
            status="success" if failed == 0 else "partial",
            message=(
                f"Successfully generated predictions for "
                f"{successful} model(s)"
            ),
            forecast_start_date=forecast_start,
            forecast_end_date=forecast_end,
            forecast_days=request.forecast_days,
            predictions=predictions,
            ensemble_prediction=ensemble_prediction,
            ensemble_dates=ensemble_dates,
            total_models_used=len(models_to_use),
            successful_predictions=successful,
            failed_predictions=failed,
            total_prediction_time_seconds=round(total_time, 3),
            created_at=datetime.now(),
        )
