"""
FastAPI application for model training
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402


def convert_predictions_to_json(predictions: Dict[str, Any]) -> Dict[str, Any]:
    json_predictions = {}
    for model, pred in predictions.items():
        if isinstance(pred, pd.Series):
            if isinstance(pred.index, pd.DatetimeIndex):
                # Convert Series with DatetimeIndex to dict of {date_str: value}
                json_predictions[model] = {
                    d.strftime("%Y-%m-%d"): v for d, v in pred.items()
                }
            else:
                # Convert other Series to a list
                json_predictions[model] = pred.tolist()
        elif isinstance(pred, pd.DataFrame) and "ds" in pred.columns:
            # Handle Prophet's DataFrame
            df = pred.copy()
            df["ds"] = df["ds"].dt.strftime("%Y-%m-%d")
            json_predictions[model] = df.to_dict(orient="records")
        elif isinstance(pred, np.ndarray):
            json_predictions[model] = pred.tolist()
        else:
            json_predictions[model] = pred
    return json_predictions

from .models import (  # noqa: E402
    ErrorResponse,
    ModelType,
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
    TrainingStatusResponse,
)
from .training_service import TrainingService  # noqa: E402
from pipeline.inference import TimeSeriesInference

# Initialize FastAPI app
app = FastAPI(
    title="Time Series Model Training and Prediction API",
    description=(
        "API for training and predicting with various time series models "
        "(ARIMA, Prophet, CatBoost, LightGBM, LSTM)"
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
training_service = TrainingService()
inference_service = TimeSeriesInference()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Time Series Model Training API",
        "version": "1.0.0",
        "docs": "/docs",
        "available_models": [model.value for model in ModelType],
        "endpoints": {
            "train": "/train",
            "status": "/status/{job_id}",
            "jobs": "/jobs",
            "health": "/health",
        },
    }


@app.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "time-series-training-api",
    }


@app.post("/train", response_model=TrainingResponse)  # type: ignore
async def train_models(request: TrainingRequest) -> TrainingResponse:
    """
    Train time series models based on the provided parameters

    - **model_types**: List of model types to train
    - **train_start_date**: Start date for training data (YYYY-MM-DD)
    - **train_end_date**: End date for training data (YYYY-MM-DD)
    - **test_start_date**: Optional start date for test data
    - **data_path**: Path to the training data file
    - **save_dir**: Directory to save trained models
    - **model_params**: Optional parameters for specific models
    """
    try:
        # Validate model types
        if not request.model_types:
            raise HTTPException(
                status_code=400,
                detail="At least one model type must be specified",
            )

        # Validate date format
        try:
            pd.to_datetime(request.train_start_date)
            pd.to_datetime(request.train_end_date)
            if request.test_start_date:
                pd.to_datetime(request.test_start_date)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid date format: {str(e)}"
            )

        # Validate date range
        if request.train_start_date >= request.train_end_date:
            raise HTTPException(
                status_code=400,
                detail="train_start_date must be before train_end_date",
            )

        # Start training
        response = await training_service.train_models(request)

        return response

    except HTTPException:
        raise
    except Exception as e:
        error_response = ErrorResponse(
            error="Training failed", detail=str(e), timestamp=datetime.now()
        )
        raise HTTPException(status_code=500, detail=error_response.dict())


@app.post("/predict", response_model=PredictionResponse)  # type: ignore
async def predict_models(request: PredictionRequest) -> PredictionResponse:
    """
    Make predictions using trained time series models

    - **model_types**: List of model types to use for prediction
    - **data_path**: Path to the data file for prediction
    - **models_dir**: Directory containing saved models
    - **test_start**: Start date for the test period
    - **forecast_days**: Number of days to forecast
    """
    try:
        # Make predictions
        predictions = inference_service.predict_all_models(
            data_path=request.data_path,
            models_dir=request.models_dir,
            test_start=request.test_start,
            forecast_days=request.forecast_days,
            model_types=request.model_types,
        )

        # Create ensemble prediction
        ensemble_prediction = None
        if len(predictions) > 1:
            ensemble_prediction = inference_service.get_ensemble_prediction(
                predictions
            )

        # Convert predictions to JSON serializable format
        json_predictions = convert_predictions_to_json(predictions)

        return PredictionResponse(
            predictions=json_predictions,
            ensemble_prediction=ensemble_prediction.tolist()
            if ensemble_prediction is not None
            else None,
            evaluation_metrics=inference_service.evaluation_metrics,
        )

    except Exception as e:
        error_response = ErrorResponse(
            error="Prediction failed", detail=str(e), timestamp=datetime.now()
        )
        raise HTTPException(status_code=500, detail=error_response.dict())


@app.get(  # type: ignore
    "/status/{job_id}", response_model=TrainingStatusResponse
)
async def get_training_status(job_id: str) -> TrainingStatusResponse:
    """
    Get the status of a training job

    - **job_id**: The job ID returned from the training endpoint
    """
    job = training_service.get_job_status(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Calculate progress percentage
    total_models = job.total_models
    completed_models = job.successful_models + job.failed_models
    progress_percentage = (
        (completed_models / total_models * 100) if total_models > 0 else 0
    )

    return TrainingStatusResponse(
        job_id=job.job_id,
        status=job.status,
        message=job.message,
        progress_percentage=progress_percentage,
        results=job.results,
        created_at=job.created_at,
        updated_at=job.completed_at or datetime.now(),
    )


@app.get("/jobs", response_model=List[TrainingResponse])
async def list_training_jobs():
    """
    List all training jobs
    """
    jobs = training_service.list_jobs()
    return jobs


@app.get("/models/available", response_model=List[str])
async def get_available_models():
    """
    Get list of available model types
    """
    return [model.value for model in ModelType]


@app.get("/models/{model_type}/info", response_model=dict)  # type: ignore
async def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get information about a specific model type
    """
    try:
        ModelType(model_type)  # Validate model type

        model_info = {
            "arima": {
                "name": "ARIMA",
                "description": (
                    "AutoRegressive Integrated Moving Average model"
                ),
                "parameters": [
                    "max_p",
                    "max_d",
                    "max_q",
                    "use_autorima",
                    "criterio",
                ],
                "output": "Time series forecast",
            },
            "prophet": {
                "name": "Prophet",
                "description": "Facebook's Prophet forecasting tool",
                "parameters": [
                    "daily_seasonality",
                    "weekly_seasonality",
                    "yearly_seasonality",
                ],
                "output": "Time series forecast with confidence intervals",
            },
            "catboost": {
                "name": "CatBoost",
                "description": "Gradient boosting on decision trees",
                "parameters": [
                    "iterations",
                    "learning_rate",
                    "depth",
                    "random_state",
                ],
                "output": "Regression predictions",
            },
            "lightgbm": {
                "name": "LightGBM",
                "description": "Light Gradient Boosting Machine",
                "parameters": [
                    "n_estimators",
                    "learning_rate",
                    "max_depth",
                    "random_state",
                ],
                "output": "Regression predictions",
            },
            "lstm": {
                "name": "LSTM",
                "description": "Long Short-Term Memory neural network",
                "parameters": [
                    "hidden_size",
                    "num_layers",
                    "num_epochs",
                    "batch_size",
                    "learning_rate",
                ],
                "output": "Time series predictions",
            },
            "all": {
                "name": "All Models",
                "description": "Train all available models",
                "parameters": "Combines all model parameters",
                "output": "Multiple model outputs",
            },
        }

        result = model_info.get(model_type, {"error": "Model type not found"})
        return result  # type: ignore

    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid model type: {model_type}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    error_response = ErrorResponse(
        error="Internal server error",
        detail=str(exc),
        timestamp=datetime.now(),
    )
    return JSONResponse(status_code=500, content=error_response.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
