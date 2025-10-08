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

#modelo para predicciones
class PredictionRequest(BaseModel):  # type: ignore
    """Request model for prediction API"""

    model_type: ModelType = Field(
        ...,
        description="Type of model to use for prediction",
        example="arima"
    )
    model_path: str = Field(
        ...,
        description="Path to the trained model file",
        example="models/arima_model.pkl"
    )
    data_path: Optional[str] = Field(
        None,
        description="Optional path to data file (for some models)",
        example=None
    )
    periods: int = Field(
        30,
        ge=1,
        le=365,
        description="Number of periods to forecast",
        example=30
    )
    prediction_params: Optional[Dict[str, Any]] = Field(
        None,
        description="Model-specific prediction parameters",
        example={
            "freq": "D",
            "sequence_length": 30
        }
    )

    #class Config:
    #    schema_extra = {
    #        "example": {
    #            "model_type": "arima",
    #            "model_path": "models/arima_model.pkl",
    #            #"data_path": "",
    #            "periods": 30,
    #            "prediction_params": {
    #                "freq": "D"
    #            }
    #        }
    #    }


class PredictionResult(BaseModel):  # type: ignore
    """Single prediction result"""

    timestamp: str = Field(
        ...,
        description="ISO format timestamp of prediction",
        example="2024-01-15T10:30:00"
    )
    date: str = Field(
        ...,
        description="Date of prediction (YYYY-MM-DD)",
        example="2024-01-15"
    )
    value: float = Field(
        ...,
        description="Predicted value",
        example=150.75
    )
    confidence_lower: Optional[float] = Field(
        None,
        description="Lower confidence interval bound",
        example=145.20
    )
    confidence_upper: Optional[float] = Field(
        None,
        description="Upper confidence interval bound", 
        example=156.30
    )
    model_type: str = Field(
        ...,
        description="Type of model used for prediction",
        example="arima"
    )


class PredictionResponse(BaseModel):  # type: ignore
    """Response model for prediction API - SINGLE MODEL"""

    prediction_id: str = Field(
        ...,
        description="Unique identifier for the prediction job",
        example="a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    )
    status: str = Field(
        ...,
        description="Status of prediction job",
        example="completed"
    )
    message: str = Field(
        ...,
        description="Descriptive message about the prediction",
        example="Prediction completed successfully for arima model"
    )
    model_type: ModelType = Field(
        ...,
        description="Type of model used for prediction"
    )
    model_path: str = Field(
        ...,
        description="Path to the model used"
    )
    periods: int = Field(
        ...,
        description="Number of periods forecasted"
    )
    predictions: List[PredictionResult] = Field(
        ...,
        description="Prediction results"
    )
    prediction_time_seconds: Optional[float] = Field(
        None,
        description="Time taken for prediction in seconds"
    )
    created_at: datetime = Field(
        ...,
        description="When the prediction job was created"
    )


class PredictionStatusResponse(BaseModel):  # type: ignore
    """Response model for prediction status check"""

    prediction_id: str
    status: str
    message: str
    progress_percentage: float
    predictions: List[PredictionResult]
    created_at: datetime
    updated_at: datetime


class ModelInfoResponse(BaseModel):  # type: ignore
    """Response model for model information"""

    model_type: str = Field(
        ...,
        description="Type of model",
        example="arima"
    )
    model_path: str = Field(
        ...,
        description="Path to the model file",
        example="models/arima_model.pkl"
    )
    model_size: Optional[str] = Field(
        None,
        description="Size of model file in MB",
        example="2.45 MB"
    )
    created_at: Optional[datetime] = Field(
        None,
        description="When the model was created"
    )
    is_available: bool = Field(
        ...,
        description="Whether the model file exists and is accessible",
        example=True
    )

class ErrorResponse(BaseModel):  # type: ignore
    """Error response model"""

    error: str
    detail: str
    timestamp: datetime
