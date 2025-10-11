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
        "data/raw/raw_stock_data.parquet",
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


class PredictionRequest(BaseModel):  # type: ignore
    """Request model for prediction API"""

    model_type: ModelType = Field(
        ...,
        description="Model type to use for prediction",
        example="arima",
    )
    forecast_days: int = Field(
        30,
        description="Number of days to forecast into the future",
        ge=1,
        le=365,
        example=30,
    )
    data_path: Optional[str] = Field(
        "data/raw/raw_stock_data.parquet",
        description="Path to the data file for context (optional for some models)",
    )
    models_dir: Optional[str] = Field(
        "models",
        description="Directory containing trained models",
    )
    include_historical: bool = Field(
        False,
        description="Whether to include historical data in response",
    )
    confidence_intervals: bool = Field(
        True,
        description="Whether to include confidence intervals (when available)",
    )


class SinglePredictionResult(BaseModel):  # type: ignore
    """Result for a single model prediction"""

    model_type: str
    status: str  # "success", "failed"
    message: str
    predictions: Optional[List[float]] = None
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    forecast_dates: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None
    model_path: Optional[str] = None


class PredictionResponse(BaseModel):  # type: ignore
    """Response model for prediction API"""

    request_id: str
    model_type: str
    status: str  # "success", "failed", "partial"
    message: str
    forecast_days: int
    result: SinglePredictionResult
    historical_data: Optional[Dict[str, List[Any]]] = None
    timestamp: datetime


class BatchPredictionRequest(BaseModel):  # type: ignore
    """Request model for batch prediction API (multiple models)"""

    model_types: List[ModelType] = Field(
        ...,
        description="List of model types to use for prediction",
        example=["arima", "prophet", "catboost"],
    )
    forecast_days: int = Field(
        30,
        description="Number of days to forecast into the future",
        ge=1,
        le=365,
        example=30,
    )
    data_path: Optional[str] = Field(
        "data/raw/raw_stock_data.parquet",
        description="Path to the data file for context",
    )
    models_dir: Optional[str] = Field(
        "models",
        description="Directory containing trained models",
    )
    include_historical: bool = Field(
        False,
        description="Whether to include historical data in response",
    )
    confidence_intervals: bool = Field(
        True,
        description="Whether to include confidence intervals (when available)",
    )


class BatchPredictionResponse(BaseModel):  # type: ignore
    """Response model for batch prediction API"""

    request_id: str
    status: str  # "success", "failed", "partial"
    message: str
    forecast_days: int
    total_models: int
    successful_models: int
    failed_models: int
    results: List[SinglePredictionResult]
    historical_data: Optional[Dict[str, List[Any]]] = None
    timestamp: datetime


class ModelInfo(BaseModel):  # type: ignore
    """Information about available models"""

    model_type: str
    model_path: str
    last_trained: Optional[str] = None
    available: bool
    file_size: Optional[str] = None


class AvailableModelsResponse(BaseModel):  # type: ignore
    """Response for available models endpoint"""

    models_dir: str
    available_models: List[ModelInfo]
    total_available: int
    timestamp: datetime


class ErrorResponse(BaseModel):  # type: ignore
    """Error response model"""

    error: str
    detail: str
    timestamp: datetime
