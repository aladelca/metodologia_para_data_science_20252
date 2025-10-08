"""
Prediction service for handling model predictions
"""

import os
import pickle
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
from catboost import CatBoostRegressor
from preprocessing.preprocess import TimeSeriesPreprocessor
from utils import LSTMModel
from .models import (
    ModelPrediction,
    ModelType,
    PredictionRequest,
    PredictionResponse,
)


class PredictionService:
    """Service class for handling predictions from trained models"""

    def __init__(self):
        self.preprocessor = TimeSeriesPreprocessor()
        self.loaded_models: Dict[str, Any] = {}

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate predictions using specified models

        Args:
            request: Prediction request with model types and parameters

        Returns:
            PredictionResponse with predictions from all models
        """
        request_id = str(uuid.uuid4())

        # Initialize response
        response = PredictionResponse(
            request_id=request_id,
            status="started",
            message="Prediction started",
            total_models=len(request.model_types),
            successful_models=0,
            failed_models=0,
            predictions=[],
            created_at=datetime.now(),
        )

        try:
            # Validate models directory exists
            if not os.path.exists(request.models_dir):
                raise FileNotFoundError(
                    f"Models directory not found: {request.models_dir}"
                )

            # Check which models require historical data
            ml_models = {
                ModelType.CATBOOST,
                ModelType.LIGHTGBM,
                ModelType.LSTM,
            }
            requires_data = (
                any(mt in ml_models for mt in request.model_types)
                or ModelType.ALL in request.model_types
            )

            # Load historical data if provided or required
            historical_data = None
            if requires_data:
                if not request.data_path:
                    raise ValueError(
                        "data_path is required for ML models "
                        "(CatBoost, LightGBM, LSTM). "
                        "Please provide the path to historical data."
                    )
                if not os.path.exists(request.data_path):
                    raise FileNotFoundError(
                        f"Data file not found: {request.data_path}"
                    )
                historical_data = self.preprocessor.load_data(
                    request.data_path
                )
            elif request.data_path and os.path.exists(request.data_path):
                historical_data = self.preprocessor.load_data(
                    request.data_path
                )

            # Generate predictions for each model type
            for model_type in request.model_types:
                if model_type == ModelType.ALL:
                    # Predict with all available models
                    all_predictions = await self._predict_all_models(
                        request, historical_data
                    )
                    response.predictions.extend(all_predictions)
                    for pred in all_predictions:
                        if pred.status == "success":
                            response.successful_models += 1
                        else:
                            response.failed_models += 1
                else:
                    prediction = await self._predict_single_model(
                        model_type, request, historical_data
                    )
                    response.predictions.append(prediction)

                    if prediction.status == "success":
                        response.successful_models += 1
                    else:
                        response.failed_models += 1

            # Update final status
            if response.successful_models > 0:
                if response.failed_models > 0:
                    response.status = "partial"
                    response.message = (
                        f"Prediction partially completed. "
                        f"{response.successful_models}/"
                        f"{response.total_models} models predicted "
                        f"successfully"
                    )
                else:
                    response.status = "success"
                    response.message = (
                        f"Prediction completed successfully for all "
                        f"{response.successful_models} models"
                    )
            else:
                response.status = "failed"
                response.message = "All model predictions failed"

        except Exception as e:
            response.status = "failed"
            response.message = f"Prediction failed: {str(e)}"

        return response

    async def _predict_single_model(
        self,
        model_type: ModelType,
        request: PredictionRequest,
        historical_data: Optional[pd.DataFrame] = None,
    ) -> ModelPrediction:
        """Predict using a single model type"""
        try:
            if model_type == ModelType.ARIMA:
                return await self._predict_arima(request, historical_data)
            elif model_type == ModelType.CATBOOST:
                return await self._predict_catboost(request, historical_data)
            elif model_type == ModelType.LSTM:
                return await self._predict_lstm(request, historical_data)
            elif model_type == ModelType.PROPHET:
                return await self._predict_prophet(request, historical_data)
            elif model_type == ModelType.LIGHTGBM:
                return await self._predict_lightgbm(request, historical_data)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            return ModelPrediction(
                model_type=model_type.value,
                status="failed",
                message=f"Failed to predict with {model_type.value}: {str(e)}",
            )

    async def _predict_arima(
        self, request: PredictionRequest,
        historical_data: Optional[pd.DataFrame]
    ) -> ModelPrediction:
        """Generate predictions using ARIMA model"""
        model_path = os.path.join(request.models_dir, "arima_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ARIMA model not found at {model_path}")

        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # nosec B301

        # Generate forecast
        forecast = model.forecast(steps=request.steps)

        # Generate prediction dates
        last_date = model.fittedvalues.index[-1]
        prediction_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=request.steps,
            freq="D",
        )

        # Get confidence intervals if available
        confidence_intervals = None
        if hasattr(model, "get_forecast"):
            forecast_result = model.get_forecast(steps=request.steps)
            conf_int = forecast_result.conf_int()
            confidence_intervals = {
                "lower": conf_int.iloc[:, 0].tolist(),
                "upper": conf_int.iloc[:, 1].tolist(),
            }

        return ModelPrediction(
            model_type=ModelType.ARIMA.value,
            status="success",
            message="ARIMA prediction completed successfully",
            predictions=(
                forecast.tolist()
                if hasattr(forecast, "tolist")
                else list(forecast)
            ),
            prediction_dates=[
                d.strftime("%Y-%m-%d") for d in prediction_dates
            ],
            confidence_intervals=confidence_intervals,
            metrics={
                "model_order": (
                    model.model_orders.get("ar", (0, 0, 0))
                    if hasattr(model, "model_orders")
                    else None
                )
            },
        )

    async def _predict_catboost(
        self,
        request: PredictionRequest,
        historical_data: Optional[pd.DataFrame],
    ) -> ModelPrediction:
        """Generate predictions using CatBoost model"""
        model_path = os.path.join(request.models_dir, "catboost_model")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"CatBoost model not found at {model_path}"
            )

        if historical_data is None:
            raise ValueError(
                "Historical data is required for CatBoost predictions"
            )

        # Load model
        model = CatBoostRegressor()
        model.load_model(model_path)

        # Prepare features for prediction
        predictions, prediction_dates = self._generate_ml_predictions(
            model, historical_data, request.steps
        )

        return ModelPrediction(
            model_type=ModelType.CATBOOST.value,
            status="success",
            message="CatBoost prediction completed successfully",
            predictions=predictions,
            prediction_dates=[
                d.strftime("%Y-%m-%d") for d in prediction_dates
            ],
            metrics={
                "feature_importance": (
                    model.get_feature_importance().tolist()
                    if hasattr(model, "get_feature_importance")
                    else None
                )
            },
        )

    async def _predict_lstm(
        self,
        request: PredictionRequest,
        historical_data: Optional[pd.DataFrame],
    ) -> ModelPrediction:
        """Generate predictions using LSTM model"""
        model_path = os.path.join(request.models_dir, "lstm_model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"LSTM model not found at {model_path}"
            )

        if historical_data is None:
            raise ValueError(
                "Historical data is required for LSTM predictions"
            )

        # Load model
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model_config = checkpoint["model_config"]

        model = LSTMModel(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            output_size=model_config["output_size"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Prepare data for prediction
        predictions, prediction_dates = self._generate_lstm_predictions(
            model, historical_data, request.steps, model_config
        )

        return ModelPrediction(
            model_type=ModelType.LSTM.value,
            status="success",
            message="LSTM prediction completed successfully",
            predictions=predictions,
            prediction_dates=[
                d.strftime("%Y-%m-%d") for d in prediction_dates
            ],
            metrics={
                "hidden_size": model_config["hidden_size"],
                "num_layers": model_config["num_layers"],
            },
        )

    async def _predict_prophet(
        self,
        request: PredictionRequest,
        historical_data: Optional[pd.DataFrame],
    ) -> ModelPrediction:
        """Generate predictions using Prophet model"""
        model_path = os.path.join(request.models_dir, "prophet_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Prophet model not found at {model_path}")

        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # nosec B301

        # Create future dataframe
        future = model.make_future_dataframe(periods=request.steps)

        # Generate forecast
        forecast = model.predict(future)

        # Get only future predictions
        predictions = forecast["yhat"].tail(request.steps).tolist()
        prediction_dates = (
            forecast["ds"]
            .tail(request.steps)
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        # Get confidence intervals
        confidence_intervals = {
            "lower": forecast["yhat_lower"].tail(request.steps).tolist(),
            "upper": forecast["yhat_upper"].tail(request.steps).tolist(),
        }

        return ModelPrediction(
            model_type=ModelType.PROPHET.value,
            status="success",
            message="Prophet prediction completed successfully",
            predictions=predictions,
            prediction_dates=prediction_dates,
            confidence_intervals=confidence_intervals,
        )

    async def _predict_lightgbm(
        self,
        request: PredictionRequest,
        historical_data: Optional[pd.DataFrame],
    ) -> ModelPrediction:
        """Generate predictions using LightGBM model"""
        model_path = os.path.join(request.models_dir, "lightgbm_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"LightGBM model not found at {model_path}"
            )

        if historical_data is None:
            raise ValueError(
                "Historical data is required for LightGBM predictions"
            )

        # Load model
        import joblib

        model = joblib.load(model_path)

        # Prepare features for prediction
        predictions, prediction_dates = self._generate_ml_predictions(
            model, historical_data, request.steps
        )

        return ModelPrediction(
            model_type=ModelType.LIGHTGBM.value,
            status="success",
            message="LightGBM prediction completed successfully",
            predictions=predictions,
            prediction_dates=[
                d.strftime("%Y-%m-%d") for d in prediction_dates
            ],
            metrics={
                "feature_importance": (
                    model.feature_importances_.tolist()
                    if hasattr(model, "feature_importances_")
                    else None
                )
            },
        )

    async def _predict_all_models(
        self,
        request: PredictionRequest,
        historical_data: Optional[pd.DataFrame],
    ) -> List[ModelPrediction]:
        """Generate predictions using all available models"""
        all_model_types = [
            ModelType.ARIMA,
            ModelType.PROPHET,
            ModelType.CATBOOST,
            ModelType.LIGHTGBM,
            ModelType.LSTM,
        ]

        predictions = []
        for model_type in all_model_types:
            prediction = await self._predict_single_model(
                model_type, request, historical_data
            )
            predictions.append(prediction)

        return predictions

    def _generate_ml_predictions(
        self,
        model: Any,
        historical_data: pd.DataFrame,
        steps: int,
    ) -> Tuple[List[float], List[pd.Timestamp]]:
        """
        Generate multi-step predictions for ML models (CatBoost, LightGBM)
        using recursive forecasting
        """
        # Prepare the most recent data for features
        train_X, train_y, _, _ = self.preprocessor.prepare_ml_data(
            historical_data
        )

        predictions = []
        last_date = historical_data.index[-1]
        prediction_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=steps, freq="D"
        )

        # Use the last available features as base
        current_features = train_X.iloc[-1:].copy()

        for i in range(steps):
            # Predict next step
            pred = model.predict(current_features)[0]
            predictions.append(float(pred))

            # Update features for next prediction (simple approach)
            # In production, you'd want more sophisticated feature engineering
            if i < steps - 1:
                # Shift lag features
                for col in current_features.columns:
                    if "lag_" in col:
                        lag_num = int(col.split("_")[-1])
                        if lag_num > 1:
                            new_col = f"lag_{lag_num - 1}"
                            if new_col in current_features.columns:
                                current_features[
                                    new_col
                                ] = current_features[
                                    col
                                ]

                # Add the prediction as the most recent value
                if "lag_1" in current_features.columns:
                    current_features["lag_1"] = pred

        return predictions, prediction_dates

    def _generate_lstm_predictions(
        self,
        model: LSTMModel,
        historical_data: pd.DataFrame,
        steps: int,
        model_config: Dict[str, Any],
    ) -> Tuple[List[float], List[pd.Timestamp]]:
        """
        Generate multi-step predictions for LSTM model
        """
        # Prepare LSTM data
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_lstm_data(
            historical_data
        )

        predictions = []
        last_date = historical_data.index[-1]
        prediction_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=steps, freq="D"
        )

        # Use the last sequence for prediction
        current_sequence = torch.FloatTensor(X_test[-1:])

        with torch.no_grad():
            for _ in range(steps):
                # Predict next step
                pred = model(current_sequence)
                pred_value = pred.item()
                predictions.append(float(pred_value))

                # Update sequence for next prediction
                # Shift the sequence and add new prediction with all features
                if current_sequence.shape[1] > 1:
                    # Get the last time step's features to use as template
                    last_features = current_sequence[:, -1:, :].clone()

                    # Create new features by repeating the last features
                    # In a more sophisticated approach, forecast all features
                    new_features = last_features.clone()

                    # Shift sequence and append new features
                    current_sequence = torch.cat(
                        [
                            current_sequence[:, 1:, :],
                            new_features,
                        ],
                        dim=1,
                    )

        return predictions, prediction_dates
