"""
Pydantic models for API requests and responses
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Available model types for training"""

    ARIMA = "arima"
    PROPHET = "prophet"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"
    LSTM = "lstm"
    ALL = "all"


class TrainingRequest(BaseModel):  # type: ignore
    """Request model for training API"""

    model_types: List[ModelType] = Field(
        ...,
        description="List of model types to train",
        example=["arima", "lstm", "catboost"],
    )
    train_start_date: str = Field(
        ...,
        description="Start date for training data (YYYY-MM-DD)",
        example="2021-01-01",
    )
    train_end_date: str = Field(
        ...,
        description="End date for training data (YYYY-MM-DD)",
        example="2023-12-31",
    )
    test_start_date: Optional[str] = Field(
        None,
        description="Start date for test data (YYYY-MM-DD)",
        example="2024-01-01",
    )
    data_path: Optional[str] = Field(
        "src/preprocessing/data/raw/raw_stock_data.parquet",
        description="Path to the training data file",
    )
    save_dir: Optional[str] = Field(
        "models", description="Directory to save trained models"
    )

    # Model-specific parameters
    arima_params: Optional[Dict[str, Any]] = Field(
        None, description="ARIMA model parameters"
    )
    prophet_params: Optional[Dict[str, Any]] = Field(
        None, description="Prophet model parameters"
    )
    catboost_params: Optional[Dict[str, Any]] = Field(
        None, description="CatBoost model parameters"
    )
    lightgbm_params: Optional[Dict[str, Any]] = Field(
        None, description="LightGBM model parameters"
    )
    lstm_params: Optional[Dict[str, Any]] = Field(
        None, description="LSTM model parameters"
    )


class ModelTrainingResult(BaseModel):  # type: ignore
    """Result for individual model training"""

    model_type: str
    status: str  # "success", "failed", "skipped"
    message: str
    training_time_seconds: Optional[float] = None
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class TrainingResponse(BaseModel):  # type: ignore
    """Response model for training API"""

    job_id: str
    status: str  # "started", "completed", "failed"
    message: str
    total_models: int
    successful_models: int
    failed_models: int
    results: List[ModelTrainingResult]
    total_training_time_seconds: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class TrainingStatusResponse(BaseModel):  # type: ignore
    """Response model for training status check"""

    job_id: str
    status: str
    message: str
    progress_percentage: float
    results: List[ModelTrainingResult]
    created_at: datetime
    updated_at: datetime


class ErrorResponse(BaseModel):  # type: ignore
    """Error response model"""

    error: str
    detail: str
    timestamp: datetime


class PredictionRequest(BaseModel):  # type: ignore
    """Request model for prediction API"""

    model_types: List[ModelType] = Field(
        ...,
        description="List of model types to use for prediction",
        example=["arima", "lstm"],
    )
    data_path: str = Field(
        "src/preprocessing/data/raw/raw_stock_data.parquet",
        description="Path to the data file for prediction",
        example="src/preprocessing/data/raw/raw_stock_data.parquet",
    )
    models_dir: str = Field(
        ...,
        description="Directory containing saved models",
        example="models",
    )
    test_start: str = Field(
        ...,
        description="Start date for the test period",
        example="2024-01-01",
    )
    forecast_days: int = Field(
        ...,
        description="Number of days to forecast",
        example=30,
    )


class PredictionResponse(BaseModel):  # type: ignore
    """Response model for prediction API"""

    predictions: Dict[str, Any] = Field(
        ...,
        description="Dictionary of predictions from all models",
    )
    ensemble_prediction: Optional[List[float]] = Field(
        None,
        description="Ensemble prediction from all models",
    )
    evaluation_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Evaluation metrics for the predictions",
    )
