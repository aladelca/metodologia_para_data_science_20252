"""Training service for handling model training operations."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.training import TimeSeriesTrainer
from src.preprocessing.preprocess import TimeSeriesPreprocessor
from src.storage.s3_utils import (
    S3Path,
    build_s3_uri,
    ensure_trailing_slash,
    is_s3_uri,
    latest_object_key,
    object_exists,
)
from src.storage.s3_utils import upload_directory as upload_directory_to_s3
from src.storage.s3_utils import upload_file as upload_file_to_s3
from src.utils import handle_timezone_compatibility

from .models import (  # isort: skip
    ModelTrainingResult,
    ModelType,
    TrainingRequest,
    TrainingResponse,
)


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


@dataclass
class TrainingStorageContext:
    """Context manager for temporary model artifacts destined for S3."""

    job_id: str
    local_dir: Path
    model_bucket: str
    model_prefix: str

    @property
    def base_prefix(self) -> Any:
        segments = [
            segment
            for segment in [
                _normalize_prefix(self.model_prefix),
                self.job_id,
            ]
            if segment
        ]
        joined = "/".join(segments)
        return ensure_trailing_slash(joined)

    def local_path(self, filename: str, subdir: Optional[str] = None) -> Path:
        target_dir = self.local_dir / subdir if subdir else self.local_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / filename

    def upload_file(
        self, local_path: Path, subdir: Optional[str] = None
    ) -> Any:
        prefix = self._prefix_for(subdir)
        key = f"{prefix}{local_path.name}"
        upload_file_to_s3(local_path, self.model_bucket, key)
        return build_s3_uri(self.model_bucket, key)

    def upload_directory(self, subdir: Optional[str] = None) -> List[str]:
        directory = self.local_dir / subdir if subdir else self.local_dir
        prefix = self._prefix_for(subdir)
        return list(
            upload_directory_to_s3(directory, self.model_bucket, prefix)
        )

    def s3_prefix(self, subdir: Optional[str] = None) -> Any:
        return self._prefix_for(subdir)

    def _prefix_for(self, subdir: Optional[str]) -> Any:
        if subdir:
            combined = f"{self.base_prefix}{_normalize_prefix(subdir)}"
            return ensure_trailing_slash(combined)
        return self.base_prefix


class TrainingService:
    """Service class for handling training operations"""

    def __init__(self):
        self.preprocessor = TimeSeriesPreprocessor()
        self.trainer = TimeSeriesTrainer(self.preprocessor)
        self.active_jobs: Dict[str, TrainingResponse] = {}
        self.data_bucket = os.environ.get(
            "TRAINING_DATA_BUCKET", "raw-data-stocks"
        )
        data_prefix_env = os.environ.get("TRAINING_DATA_PREFIX", "stock_data")
        self.data_prefix = (
            ensure_trailing_slash(_normalize_prefix(data_prefix_env))
            if data_prefix_env
            else ""
        )
        self.model_bucket = os.environ.get(
            "TRAINING_MODEL_BUCKET", "raw-data-stocks"
        )
        model_prefix_env = os.environ.get("TRAINING_MODEL_PREFIX", "models")
        self.model_prefix = (
            ensure_trailing_slash(_normalize_prefix(model_prefix_env))
            if model_prefix_env
            else ""
        )

    async def train_models(self, request: TrainingRequest) -> TrainingResponse:
        """Execute model training using datasets stored in S3."""

        job_id = str(uuid.uuid4())
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
            data_uri = self._resolve_data_source(request.data_path)

            target_bucket, target_prefix = self._resolve_storage_target(
                request.save_dir
            )

            with TemporaryDirectory(prefix=f"training-{job_id}-") as temp_dir:
                storage = TrainingStorageContext(
                    job_id=job_id,
                    local_dir=Path(temp_dir),
                    model_bucket=target_bucket,
                    model_prefix=target_prefix,
                )

                data = self.preprocessor.load_data(data_uri)
                filtered_data = self._filter_data_by_dates(
                    data, request.train_start_date, request.train_end_date
                )

                if filtered_data.empty:
                    raise ValueError(
                        "No data available for the specified date range"
                    )

                for model_type in request.model_types:
                    result = await self._train_single_model(
                        model_type,
                        filtered_data,
                        request,
                        storage,
                        data_uri,
                    )
                    response.results.append(result)

                    if result.status == "success":
                        response.successful_models += 1
                    else:
                        response.failed_models += 1

            response.status = "completed"
            response.message = (
                "Training completed. "
                f"{response.successful_models}/{response.total_models} "
                "models trained successfully"
            )
            response.completed_at = datetime.now()

            if response.results:
                response.total_training_time_seconds = sum(
                    r.training_time_seconds or 0 for r in response.results
                )

        except Exception as exc:
            response.status = "failed"
            response.message = f"Training failed: {str(exc)}"
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

    def _resolve_data_source(self, data_path: Optional[str]) -> Any:
        """Resolve the concrete dataset location, preferring S3 prefixes."""

        if data_path:
            source = data_path
        else:
            if not self.data_prefix:
                raise ValueError(
                    "Configure TRAINING_DATA_PREFIX or provide data_path"
                )
            source = build_s3_uri(self.data_bucket, self.data_prefix)

        if is_s3_uri(source):
            s3_path = S3Path.from_uri(source)
            s3_location = build_s3_uri(s3_path.bucket, s3_path.key)
            if source.endswith("/"):
                key = latest_object_key(s3_path.bucket, s3_path.key)
                if not key:
                    raise FileNotFoundError(
                        f"No objects found under {s3_location}"
                    )
                return build_s3_uri(s3_path.bucket, key)

            if not object_exists(s3_path.bucket, s3_path.key):
                raise FileNotFoundError(f"S3 object not found: {s3_location}")
            return source

        if not os.path.exists(source):
            raise FileNotFoundError(f"Data file not found: {source}")

        return source

    def _resolve_storage_target(
        self, save_dir: Optional[str]
    ) -> Tuple[str, str]:
        """Determine where to upload trained model artifacts."""

        if save_dir:
            if is_s3_uri(save_dir):
                normalized_uri = ensure_trailing_slash(save_dir)
                path = S3Path.from_uri(normalized_uri)
                return path.bucket, ensure_trailing_slash(path.key)
            raise ValueError(
                "save_dir must be an S3 URI when running inside the training "
                "service"
            )

        return self.model_bucket, self.model_prefix

    async def _train_single_model(
        self,
        model_type: ModelType,
        data: pd.DataFrame,
        request: TrainingRequest,
        storage: TrainingStorageContext,
        data_uri: str,
    ) -> ModelTrainingResult:
        """Train a single model type"""
        start_time = datetime.now()

        try:
            if model_type == ModelType.ARIMA:
                result = await self._train_arima(data, request, storage)
            elif model_type == ModelType.PROPHET:
                result = await self._train_prophet(data, request, storage)
            elif model_type == ModelType.CATBOOST:
                result = await self._train_catboost(data, request, storage)
            elif model_type == ModelType.LIGHTGBM:
                result = await self._train_lightgbm(data, request, storage)
            elif model_type == ModelType.LSTM:
                result = await self._train_lstm(data, request, storage)
            elif model_type == ModelType.ALL:
                result = await self._train_all_models(
                    data_uri, request, storage
                )
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
        self,
        data: pd.DataFrame,
        request: TrainingRequest,
        storage: TrainingStorageContext,
    ) -> Dict[str, Any]:
        """Train ARIMA model"""
        arima_data = self.preprocessor.prepare_arima_data(data)
        local_path = storage.local_path("arima_model.pkl")

        model, best_order, results_df = self.trainer.train_arima(
            arima_data,
            save_path=str(local_path),
            **(request.arima_params or {}),
        )

        model_uri = storage.upload_file(local_path)

        return {
            "model_path": model_uri,
            "metrics": {
                "best_order": best_order,
                "aic": model.aic,
                "bic": model.bic,
            },
        }

    async def _train_prophet(
        self,
        data: pd.DataFrame,
        request: TrainingRequest,
        storage: TrainingStorageContext,
    ) -> Dict[str, Any]:
        """Train Prophet model"""
        prophet_data, exog_vars = self.preprocessor.prepare_prophet_data(data)
        local_path = storage.local_path("prophet_model.pkl")

        model = self.trainer.train_prophet(
            prophet_data,
            exog_vars=exog_vars,
            save_path=str(local_path),
            **(request.prophet_params or {}),
        )

        model_uri = storage.upload_file(local_path)

        return {
            "model_path": model_uri,
            "metrics": {
                "exog_vars": exog_vars,
                "components": list(model.params.keys())
                if hasattr(model, "params")
                else [],
            },
        }

    async def _train_catboost(
        self,
        data: pd.DataFrame,
        request: TrainingRequest,
        storage: TrainingStorageContext,
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
                "No training data available. "
                f"Train X: {len(train_X)}, Train y: {len(train_y)}"
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

        # Additional validation for CatBoost
        if hasattr(train_y, "dtype") and train_y.dtype == "object":
            raise ValueError(
                "Target variable has object dtype, expected numeric. "
                f"Unique values: {train_y.unique()[:5]}"
            )

        if train_y.nunique() < 2:
            raise ValueError(
                "Target variable has only "
                f"{train_y.nunique()} unique values, need at least 2 "
                "for regression"
            )

        local_path = storage.local_path("catboost_model.cbm")

        model = self.trainer.train_catboost(
            train_X,
            train_y,
            test_X,
            test_y,
            save_path=str(local_path),
            **(request.catboost_params or {}),
        )

        model_uri = storage.upload_file(local_path)

        return {
            "model_path": model_uri,
            "metrics": {
                "best_iteration": getattr(model, "best_iteration_", None),
                "feature_importance": model.feature_importances_.tolist(),
                "training_samples": len(train_X),
                "test_samples": len(test_X) if test_X is not None else 0,
            },
        }

    async def _train_lightgbm(
        self,
        data: pd.DataFrame,
        request: TrainingRequest,
        storage: TrainingStorageContext,
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
                "No training data available. "
                f"Train X: {len(train_X)}, Train y: {len(train_y)}"
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

        local_path = storage.local_path("lightgbm_model.pkl")

        model = self.trainer.train_lightgbm(
            train_X,
            train_y,
            test_X,
            test_y,
            save_path=str(local_path),
            **(request.lightgbm_params or {}),
        )

        model_uri = storage.upload_file(local_path)

        return {
            "model_path": model_uri,
            "metrics": {
                "best_iteration": getattr(model, "best_iteration_", None),
                "feature_importance": model.feature_importances_.tolist(),
                "training_samples": len(train_X),
                "test_samples": len(test_X) if test_X is not None else 0,
            },
        }

    async def _train_lstm(
        self,
        data: pd.DataFrame,
        request: TrainingRequest,
        storage: TrainingStorageContext,
    ) -> Dict[str, Any]:
        """Train LSTM model"""
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_lstm_data(
            data
        )

        local_path = storage.local_path("lstm_model.pth")

        model, train_losses, val_losses = self.trainer.train_lstm(
            X_train,
            y_train,
            X_test,
            y_test,
            save_path=str(local_path),
            **(request.lstm_params or {}),
        )

        model_uri = storage.upload_file(local_path)

        return {
            "model_path": model_uri,
            "metrics": {
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "num_epochs": len(train_losses),
            },
        }

    async def _train_all_models(
        self,
        data_uri: str,
        request: TrainingRequest,
        storage: TrainingStorageContext,
    ) -> Dict[str, Any]:
        """Train all available models"""
        local_dir = storage.local_dir / "all_models"
        local_dir.mkdir(parents=True, exist_ok=True)
        models = self.trainer.train_all_models(
            data_uri,
            save_dir=str(local_dir),
            arima=request.arima_params or {},
            prophet=request.prophet_params or {},
            catboost=request.catboost_params or {},
            lightgbm=request.lightgbm_params or {},
            lstm=request.lstm_params or {},
        )

        uploaded = storage.upload_directory(subdir="all_models")

        return {
            "model_path": build_s3_uri(
                storage.model_bucket,
                storage.s3_prefix("all_models"),
            ),
            "metrics": {
                "trained_models": list(models.keys()),
                "summary": self.trainer.get_training_summary(),
                "artifacts": uploaded,
            },
        }

    def get_job_status(self, job_id: str) -> Optional[TrainingResponse]:
        """Get the status of a training job"""
        return self.active_jobs.get(job_id)

    def list_jobs(self) -> List[TrainingResponse]:
        """List all training jobs"""
        return list(self.active_jobs.values())
