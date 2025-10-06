"""
Training service for handling model training operations
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from pipeline.training import TimeSeriesTrainer
from preprocessing.preprocess import TimeSeriesPreprocessor
from utils import handle_timezone_compatibility

from .models import (  # noqa: E501
    ModelTrainingResult,
    ModelType,
    TrainingRequest,
    TrainingResponse,
)


class TrainingService:
    """Service class for handling training operations"""

    def __init__(self):
        self.preprocessor = TimeSeriesPreprocessor()
        self.trainer = TimeSeriesTrainer(self.preprocessor)
        self.active_jobs: Dict[str, TrainingResponse] = {}

    async def train_models(self, request: TrainingRequest) -> TrainingResponse:
        """
        Train models based on the request parameters

        Args:
            request: Training request with model types and parameters

        Returns:
            TrainingResponse with results
        """
        job_id = str(uuid.uuid4())

        # Initialize response
        response = TrainingResponse(
            job_id=job_id,
            status="started",
            message="Training started",
            total_models=len(request.model_types),
            successful_models=0,
            failed_models=0,
            results=[],
            created_at=datetime.now(),
        )

        self.active_jobs[job_id] = response

        try:
            # Validate data file exists
            if not os.path.exists(request.data_path):
                raise FileNotFoundError(
                    f"Data file not found: {request.data_path}"
                )

            # Create save directory
            os.makedirs(request.save_dir, exist_ok=True)

            # Load and validate data
            data = self.preprocessor.load_data(request.data_path)

            # Filter data by date range
            filtered_data = self._filter_data_by_dates(
                data, request.train_start_date, request.train_end_date
            )

            if filtered_data.empty:
                raise ValueError(
                    "No data available for the specified date range"
                )

            # Train each model type
            for model_type in request.model_types:
                result = await self._train_single_model(
                    model_type, filtered_data, request
                )
                response.results.append(result)

                if result.status == "success":
                    response.successful_models += 1
                else:
                    response.failed_models += 1

            # Update final status
            response.status = "completed"
            response.message = (
                f"Training completed. {response.successful_models}/"
                f"{response.total_models} models trained successfully"
            )
            response.completed_at = datetime.now()

            if response.results:
                response.total_training_time_seconds = sum(
                    r.training_time_seconds or 0 for r in response.results
                )

        except Exception as e:
            response.status = "failed"
            response.message = f"Training failed: {str(e)}"
            response.completed_at = datetime.now()

        return response

    def _filter_data_by_dates(
        self, data: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Filter data by date range"""
        # Use utility function to handle timezone compatibility
        start_dt = handle_timezone_compatibility(start_date, data.index)
        end_dt = handle_timezone_compatibility(end_date, data.index)

        return data[(data.index >= start_dt) & (data.index <= end_dt)]

    async def _train_single_model(
        self,
        model_type: ModelType,
        data: pd.DataFrame,
        request: TrainingRequest,
    ) -> ModelTrainingResult:
        """Train a single model type"""
        start_time = datetime.now()

        try:
            if model_type == ModelType.ARIMA:
                result = await self._train_arima(data, request)
            elif model_type == ModelType.PROPHET:
                result = await self._train_prophet(data, request)
            elif model_type == ModelType.CATBOOST:
                result = await self._train_catboost(data, request)
            elif model_type == ModelType.LIGHTGBM:
                result = await self._train_lightgbm(data, request)
            elif model_type == ModelType.LSTM:
                result = await self._train_lstm(data, request)
            elif model_type == ModelType.ALL:
                result = await self._train_all_models(data, request)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            training_time = (datetime.now() - start_time).total_seconds()

            return ModelTrainingResult(
                model_type=model_type.value,
                status="success",
                message=f"{model_type.value} model trained successfully",
                training_time_seconds=training_time,
                model_path=result.get("model_path"),
                metrics=result.get("metrics"),
            )

        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            return ModelTrainingResult(
                model_type=model_type.value,
                status="failed",
                message=f"Failed to train {model_type.value}: {str(e)}",
                training_time_seconds=training_time,
            )

    async def _train_arima(
        self, data: pd.DataFrame, request: TrainingRequest
    ) -> Dict[str, Any]:
        """Train ARIMA model"""
        arima_data = self.preprocessor.prepare_arima_data(data)

        model_path = os.path.join(request.save_dir, "arima_model.pkl")

        model, best_order, results_df = self.trainer.train_arima(
            arima_data, save_path=model_path, **(request.arima_params or {})
        )

        return {
            "model_path": model_path,
            "metrics": {
                "best_order": best_order,
                "aic": model.aic,
                "bic": model.bic,
            },
        }

    async def _train_prophet(
        self, data: pd.DataFrame, request: TrainingRequest
    ) -> Dict[str, Any]:
        """Train Prophet model"""
        prophet_data, exog_vars = self.preprocessor.prepare_prophet_data(data)

        model_path = os.path.join(request.save_dir, "prophet_model.pkl")

        model = self.trainer.train_prophet(
            prophet_data,
            exog_vars=exog_vars,
            save_path=model_path,
            **(request.prophet_params or {}),
        )

        return {
            "model_path": model_path,
            "metrics": {
                "exog_vars": exog_vars,
                "components": list(model.params.keys())
                if hasattr(model, "params")
                else [],
            },
        }

    async def _train_catboost(
        self, data: pd.DataFrame, request: TrainingRequest
    ) -> Dict[str, Any]:
        """Train CatBoost model"""
        train_X, train_y, test_X, test_y = self.preprocessor.prepare_ml_data(
            data,
            train_start=request.train_start_date,
            train_end=request.train_end_date,
            test_start=request.test_start_date,
        )

        # Validate data before training
        if len(train_X) == 0 or len(train_y) == 0:
            raise ValueError(
                f"No training data available. Train X: {len(train_X)}, "
                f"Train y: {len(train_y)}"
            )

        if train_y.isna().all():
            raise ValueError("All target values are NaN after preprocessing")

        # Remove any remaining NaN values
        valid_indices = ~train_y.isna()
        train_X = train_X[valid_indices]
        train_y = train_y[valid_indices]
        print(train_X)
        if len(train_X) == 0:
            raise ValueError(
                "No valid training data after removing NaN values"
            )

        # Additional validation for CatBoost
        if hasattr(train_y, 'dtype') and train_y.dtype == 'object':
            raise ValueError(
                f"Target variable has object dtype, expected numeric. "
                f"Unique values: {train_y.unique()[:5]}"
            )

        if train_y.nunique() < 2:
            raise ValueError(
                f"Target variable has only {train_y.nunique()} unique values, "
                f"need at least 2 for regression"
            )

        model_path = os.path.join(request.save_dir, "catboost_model")

        model = self.trainer.train_catboost(
            train_X,
            train_y,
            test_X,
            test_y,
            save_path=model_path,
            **(request.catboost_params or {}),
        )

        return {
            "model_path": model_path,
            "metrics": {
                "best_iteration": getattr(model, "best_iteration_", None),
                "feature_importance": model.feature_importances_.tolist(),
                "training_samples": len(train_X),
                "test_samples": len(test_X) if test_X is not None else 0,
            },
        }

    async def _train_lightgbm(
        self, data: pd.DataFrame, request: TrainingRequest
    ) -> Dict[str, Any]:
        """Train LightGBM model"""
        train_X, train_y, test_X, test_y = self.preprocessor.prepare_ml_data(
            data,
            train_start=request.train_start_date,
            train_end=request.train_end_date,
            test_start=request.test_start_date,
        )

        # Validate data before training
        if len(train_X) == 0 or len(train_y) == 0:
            raise ValueError(
                f"No training data available. Train X: {len(train_X)}, "
                f"Train y: {len(train_y)}"
            )

        if train_y.isna().all():
            raise ValueError("All target values are NaN after preprocessing")

        # Remove any remaining NaN values
        valid_indices = ~train_y.isna()
        train_X = train_X[valid_indices]
        train_y = train_y[valid_indices]

        if len(train_X) == 0:
            raise ValueError(
                "No valid training data after removing NaN values"
            )

        model_path = os.path.join(request.save_dir, "lightgbm_model.pkl")

        model = self.trainer.train_lightgbm(
            train_X,
            train_y,
            test_X,
            test_y,
            save_path=model_path,
            **(request.lightgbm_params or {}),
        )

        return {
            "model_path": model_path,
            "metrics": {
                "best_iteration": getattr(model, "best_iteration_", None),
                "feature_importance": model.feature_importances_.tolist(),
                "training_samples": len(train_X),
                "test_samples": len(test_X) if test_X is not None else 0,
            },
        }

    async def _train_lstm(
        self, data: pd.DataFrame, request: TrainingRequest
    ) -> Dict[str, Any]:
        """Train LSTM model"""
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_lstm_data(
            data
        )

        model_path = os.path.join(request.save_dir, "lstm_model.pth")

        model, train_losses, val_losses = self.trainer.train_lstm(
            X_train,
            y_train,
            X_test,
            y_test,
            save_path=model_path,
            **(request.lstm_params or {}),
        )

        return {
            "model_path": model_path,
            "metrics": {
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "num_epochs": len(train_losses),
            },
        }

    async def _train_all_models(
        self, data: pd.DataFrame, request: TrainingRequest
    ) -> Dict[str, Any]:
        """Train all available models"""
        models = self.trainer.train_all_models(
            request.data_path,
            save_dir=request.save_dir,
            arima=request.arima_params or {},
            prophet=request.prophet_params or {},
            catboost=request.catboost_params or {},
            lightgbm=request.lightgbm_params or {},
            lstm=request.lstm_params or {},
        )

        return {
            "model_path": request.save_dir,
            "metrics": {
                "trained_models": list(models.keys()),
                "summary": self.trainer.get_training_summary(),
            },
        }

    def get_job_status(self, job_id: str) -> Optional[TrainingResponse]:
        """Get the status of a training job"""
        return self.active_jobs.get(job_id)

    def list_jobs(self) -> List[TrainingResponse]:
        """List all training jobs"""
        return list(self.active_jobs.values())
