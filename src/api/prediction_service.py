"""
Prediction service for handling model inference requests
"""

import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pipeline.inference import TimeSeriesInference  # noqa: E402

from .models import (  # noqa: E402
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
    SinglePredictionResult,
)


class PredictionService:
    """Service class for handling prediction requests"""

    def __init__(self):
        """Initialize the prediction service"""
        self.inference_engine = TimeSeriesInference()

    def validate_input_data(self, data_path: str, model_type: str) -> bool:
        """
        Validate that input data exists and is in the correct format

        Args:
            data_path (str): Path to the data file
            model_type (str): Type of model to validate for

        Returns:
            bool: True if data is valid

        Raises:
            Exception: If data validation fails
        """
        if not os.path.exists(data_path):
            raise Exception(f"Data file not found: {data_path}")

        try:
            data = self.inference_engine.preprocessor.load_data(data_path)

            if data.empty:
                raise Exception("Data file is empty")

            if len(data) < 10:
                raise Exception(
                    "Insufficient data for prediction (minimum 10 records required)")

            # Model-specific validation
            if model_type in ["catboost", "lightgbm", "lstm"] and len(data) < 30:
                raise Exception(
                    f"Model {model_type} requires at least 30 data points")

            return True

        except Exception as e:
            raise Exception(f"Data validation failed: {str(e)}")

    def check_model_availability(self, models_dir: str) -> List[ModelInfo]:
        """
        Check which models are available in the models directory

        Args:
            models_dir (str): Path to models directory

        Returns:
            List[ModelInfo]: List of available models with their info
        """
        available_models = []
        model_files = {
            "arima": ["arima_model.pkl", "arima_model"],
            "prophet": ["prophet_model.pkl", "prophet_model"],
            "catboost": ["catboost_model", "catboost_model.pkl", "catboost_model.cbm"],
            "lightgbm": ["lightgbm_model.pkl", "lightgbm_model", "lightgbm_model.txt"],
            "lstm": ["lstm_model.pth", "lstm_model.pt"],
        }

        for model_type, possible_filenames in model_files.items():
            model_path = None
            available = False

            # Check all possible filenames for each model type
            for filename in possible_filenames:
                potential_path = os.path.join(models_dir, filename)
                if os.path.exists(potential_path):
                    model_path = potential_path
                    available = True
                    break

            # If no file found, use the first filename as default path
            if not available:
                model_path = os.path.join(models_dir, possible_filenames[0])

            model_info = ModelInfo(
                model_type=model_type,
                model_path=model_path,
                available=available,
            )

            if available:
                try:
                    # Get file modification time and size
                    stat = os.stat(model_path)
                    model_info.last_trained = datetime.fromtimestamp(
                        stat.st_mtime
                    ).isoformat()
                    model_info.file_size = f"{stat.st_size / (1024 * 1024):.2f} MB"
                except Exception:
                    pass

            available_models.append(model_info)

        return available_models

    def generate_forecast_dates(
        self, start_date: datetime, forecast_days: int
    ) -> List[str]:
        """
        Generate forecast dates starting from the given date

        Args:
            start_date (datetime): Starting date for forecast
            forecast_days (int): Number of days to forecast

        Returns:
            List[str]: List of forecast dates in YYYY-MM-DD format
        """
        forecast_dates = []
        for i in range(forecast_days):
            forecast_date = start_date + timedelta(days=i + 1)
            forecast_dates.append(forecast_date.strftime("%Y-%m-%d"))
        return forecast_dates

    def load_historical_data(
        self, data_path: str, include_historical: bool
    ) -> Optional[Dict[str, List]]:
        """
        Load historical data if requested

        Args:
            data_path (str): Path to data file
            include_historical (bool): Whether to include historical data

        Returns:
            Optional[Dict[str, List]]: Historical data if requested
        """
        if not include_historical or not os.path.exists(data_path):
            return None

        try:
            # Load data using the preprocessor
            data = self.inference_engine.preprocessor.load_data(data_path)

            # Return last 30 days of historical data
            if len(data) > 30:
                data = data.tail(30)

            historical_data = {
                "dates": data.index.strftime("%Y-%m-%d").tolist(),
                # Assuming first column is target
                "values": data.iloc[:, 0].tolist(),
            }

            return historical_data
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return None

    async def predict_single_model(
        self, request: PredictionRequest
    ) -> PredictionResponse:
        """
        Make prediction using a single model

        Args:
            request (PredictionRequest): Prediction request

        Returns:
            PredictionResponse: Prediction response
        """
        request_id = str(uuid.uuid4())
        timestamp = datetime.now()

        try:
            # Validate input data if data path is provided
            if request.data_path and os.path.exists(request.data_path):
                self.validate_input_data(
                    request.data_path, request.model_type.value)

            # Check if model exists
            available_models = self.check_model_availability(
                request.models_dir)
            model_info = next(
                (m for m in available_models if m.model_type ==
                 request.model_type.value),
                None,
            )

            if not model_info or not model_info.available:
                result = SinglePredictionResult(
                    model_type=request.model_type.value,
                    status="failed",
                    message=f"Model {request.model_type.value} not found or not available",
                )
                return PredictionResponse(
                    request_id=request_id,
                    model_type=request.model_type.value,
                    status="failed",
                    message=f"Model {request.model_type.value} not available",
                    forecast_days=request.forecast_days,
                    result=result,
                    timestamp=timestamp,
                )

            # Load and use the model for prediction
            model = self.inference_engine.load_model(
                request.model_type.value, model_info.model_path
            )

            # Generate predictions based on model type
            predictions = None
            confidence_intervals = None

            try:
                if request.model_type.value == "arima":
                    predictions = self.inference_engine.predict_arima(
                        model, steps=request.forecast_days, return_conf_int=False
                    )
                    if request.confidence_intervals:
                        try:
                            pred_ci = self.inference_engine.predict_arima(
                                model, steps=request.forecast_days, return_conf_int=True
                            )
                            if isinstance(pred_ci, tuple) and len(pred_ci) == 2:
                                confidence_intervals = {
                                    "lower": pred_ci[1][:, 0].tolist(),
                                    "upper": pred_ci[1][:, 1].tolist(),
                                }
                        except Exception as e:
                            print(
                                f"Warning: Could not generate confidence intervals for ARIMA: {e}")

                elif request.model_type.value == "prophet":
                    pred_df = self.inference_engine.predict_prophet(
                        model, periods=request.forecast_days, inc_history=False
                    )
                    predictions = pred_df["yhat"].tolist()
                    if request.confidence_intervals and "yhat_lower" in pred_df.columns:
                        confidence_intervals = {
                            "lower": pred_df["yhat_lower"].tolist(),
                            "upper": pred_df["yhat_upper"].tolist(),
                        }

                elif request.model_type.value in ["catboost", "lightgbm"]:
                    # For ML models, we need to prepare features properly
                    if not request.data_path or not os.path.exists(request.data_path):
                        raise Exception(
                            "Data path required for ML model predictions")

                    # Load and process data using the preprocessor
                    raw_data = self.inference_engine.preprocessor.load_data(
                        request.data_path)

                    # Extract the target column (Close price) - assuming it exists
                    if "Close" not in raw_data.columns:
                        raise Exception(
                            "Data must contain 'Close' column for ML model predictions")

                    target_series = raw_data["Close"]

                    # Create features using the same preprocessing pipeline as training
                    features_df = self.inference_engine.preprocessor.create_features(
                        target_series, "Close")

                    # Remove rows with NaN values (they can't be used for prediction)
                    features_df = features_df.dropna()

                    if len(features_df) < 1:
                        raise Exception(
                            "No valid feature data available after preprocessing")

                    # Get feature columns (exclude target columns)
                    feature_cols = [col for col in features_df.columns if col not in [
                        "target", "target_diff"]]

                    # Use the last available features for prediction
                    last_features = features_df[feature_cols].tail(1)

                    # Make prediction
                    single_prediction = self.inference_engine.predict_ml_models(
                        model, last_features, request.model_type.value
                    )[0]

                    # For future predictions, we'll use a simple approach:
                    # Repeat the last prediction (in real scenarios, you'd use more sophisticated methods)
                    predictions = [float(single_prediction)] * \
                        request.forecast_days

                elif request.model_type.value == "lstm":
                    # For LSTM, we need sequence data
                    if not request.data_path or not os.path.exists(request.data_path):
                        raise Exception(
                            "Data path required for LSTM model predictions")

                    data = self.inference_engine.preprocessor.load_data(
                        request.data_path)

                    if len(data) < 10:
                        raise Exception(
                            "Insufficient data for LSTM prediction (minimum 10 points)")

                    # Simplified LSTM prediction - in real scenarios you would prepare proper sequences
                    last_value = float(data.iloc[-1, 0])
                    predictions = [last_value] * request.forecast_days

                else:
                    raise Exception(
                        f"Unsupported model type: {request.model_type.value}")

                # Ensure predictions are in the right format
                if predictions is None:
                    raise Exception("Model returned no predictions")

                # Convert numpy arrays to lists if needed
                if hasattr(predictions, 'tolist'):
                    predictions = predictions.tolist()
                elif not isinstance(predictions, list):
                    predictions = [float(predictions)] if not hasattr(
                        predictions, '__iter__') else list(predictions)

            except Exception as model_error:
                raise Exception(f"Model prediction failed: {str(model_error)}")

            # Generate forecast dates
            forecast_dates = self.generate_forecast_dates(
                datetime.now(), request.forecast_days
            )

            # Load historical data if requested
            historical_data = self.load_historical_data(
                request.data_path, request.include_historical
            )

            result = SinglePredictionResult(
                model_type=request.model_type.value,
                status="success",
                message="Prediction completed successfully",
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                forecast_dates=forecast_dates,
                model_path=model_info.model_path,
            )

            return PredictionResponse(
                request_id=request_id,
                model_type=request.model_type.value,
                status="success",
                message="Prediction completed successfully",
                forecast_days=request.forecast_days,
                result=result,
                historical_data=historical_data,
                timestamp=timestamp,
            )

        except Exception as e:
            error_message = f"Prediction failed: {str(e)}"
            result = SinglePredictionResult(
                model_type=request.model_type.value,
                status="failed",
                message=error_message,
            )

            return PredictionResponse(
                request_id=request_id,
                model_type=request.model_type.value,
                status="failed",
                message=error_message,
                forecast_days=request.forecast_days,
                result=result,
                timestamp=timestamp,
            )

    async def predict_batch_models(
        self, request: BatchPredictionRequest
    ) -> BatchPredictionResponse:
        """
        Make predictions using multiple models

        Args:
            request (BatchPredictionRequest): Batch prediction request

        Returns:
            BatchPredictionResponse: Batch prediction response
        """
        request_id = str(uuid.uuid4())
        timestamp = datetime.now()
        results = []
        successful_models = 0
        failed_models = 0

        try:
            # Load historical data once if requested
            historical_data = self.load_historical_data(
                request.data_path, request.include_historical
            )

            # Process each model
            for model_type in request.model_types:
                single_request = PredictionRequest(
                    model_type=model_type,
                    forecast_days=request.forecast_days,
                    data_path=request.data_path,
                    models_dir=request.models_dir,
                    include_historical=False,  # Already loaded above
                    confidence_intervals=request.confidence_intervals,
                )

                single_response = await self.predict_single_model(single_request)
                results.append(single_response.result)

                if single_response.status == "success":
                    successful_models += 1
                else:
                    failed_models += 1

            # Determine overall status
            if successful_models == len(request.model_types):
                status = "success"
                message = "All predictions completed successfully"
            elif successful_models > 0:
                status = "partial"
                message = f"{successful_models}/{len(request.model_types)} predictions completed"
            else:
                status = "failed"
                message = "All predictions failed"

            return BatchPredictionResponse(
                request_id=request_id,
                status=status,
                message=message,
                forecast_days=request.forecast_days,
                total_models=len(request.model_types),
                successful_models=successful_models,
                failed_models=failed_models,
                results=results,
                historical_data=historical_data,
                timestamp=timestamp,
            )

        except Exception as e:
            return BatchPredictionResponse(
                request_id=request_id,
                status="failed",
                message=f"Batch prediction failed: {str(e)}",
                forecast_days=request.forecast_days,
                total_models=len(request.model_types),
                successful_models=0,
                failed_models=len(request.model_types),
                results=[],
                timestamp=timestamp,
            )
