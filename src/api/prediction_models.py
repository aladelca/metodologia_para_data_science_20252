"""
Pydantic models for Prediction API requests and responses
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for prediction API"""

    model_name: Optional[str] = Field(
        None,
        description=(
            "Name of the model to use for prediction. "
            "If not provided, all available models will be used."
        ),
        example="lstm_model",
    )
    forecast_days: int = Field(
        30,
        description="Number of days to forecast into the future",
        ge=1,
        le=365,
        example=30,
    )
    data_path: Optional[str] = Field(
        None,
        description=(
            "Path to historical data for prediction. "
            "If not provided, uses default path"
        ),
        example="data/raw/raw_stock_data.parquet",
    )
    include_confidence_intervals: bool = Field(
        True,
        description="Whether to include confidence intervals (if supported)",
    )
    ensemble_method: Optional[str] = Field(
        "mean",
        description=(
            "Ensemble method when using all models: "
            "'mean', 'median', or 'weighted'"
        ),
        example="mean",
    )


class ModelPrediction(BaseModel):
    """Individual model prediction result"""

    model_name: str
    model_type: str
    predictions: List[float]
    dates: List[str]
    confidence_lower: Optional[List[float]] = None
    confidence_upper: Optional[List[float]] = None
    metrics: Optional[Dict[str, float]] = None
    prediction_time_seconds: float


class PredictionResponse(BaseModel):
    """Response model for prediction API"""

    request_id: str
    status: str  # "success", "partial", "failed"
    message: str
    forecast_start_date: str
    forecast_end_date: str
    forecast_days: int
    predictions: List[ModelPrediction]
    ensemble_prediction: Optional[List[float]] = None
    ensemble_dates: Optional[List[str]] = None
    total_models_used: int
    successful_predictions: int
    failed_predictions: int
    total_prediction_time_seconds: float
    created_at: datetime


class AvailableModelsResponse(BaseModel):
    """Response model for available models endpoint"""

    models_directory: str
    available_models: List[str]
    model_details: List[Dict[str, Any]]
    total_models: int


class PredictionErrorResponse(BaseModel):
    """Error response model for predictions"""

    error: str
    detail: str
    request_id: str
    timestamp: datetime
