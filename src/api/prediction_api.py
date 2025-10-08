"""
FastAPI application for model predictions
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from .prediction_models import (  # noqa: E402
    AvailableModelsResponse,
    PredictionErrorResponse,
    PredictionRequest,
    PredictionResponse,
)
from .prediction_service import PredictionService  # noqa: E402

# Initialize FastAPI app
app = FastAPI(
    title="Time Series Prediction API",
    description=(
        "API for making predictions using trained time series models "
        "(ARIMA, Prophet, CatBoost, LightGBM, LSTM)"
    ),
    version="1.0.0",
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

# Initialize prediction service
prediction_service = PredictionService()


@app.get("/", response_model=dict)
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    return {
        "message": "Time Series Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "/predict",
            "models": "/models",
            "health": "/health",
        },
        "description": (
            "Use this API to make predictions with trained time series models"
        ),
    }


@app.get("/health", response_model=dict)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "time-series-prediction-api",
        "models_directory": prediction_service.models_dir,
    }


@app.get("/models", response_model=AvailableModelsResponse)
async def get_available_models() -> AvailableModelsResponse:
    """
    Get list of all available trained models

    Returns information about each model including:
    - Model name
    - Model type
    - File path
    - File size
    - Last modified date
    """
    try:
        response = prediction_service.get_available_models()
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving models: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make predictions using trained models

    ### Parameters:
    - **model_name** (optional): Name of specific model to use.
      If not provided, all available models will be used.
      Examples: 'lstm_model.pth', 'catboost_model', 'prophet_model.pkl'

    - **forecast_days**: Number of days to forecast (1-365, default: 30)

    - **data_path** (optional): Path to historical data file.
      If not provided, uses default data path.

    - **include_confidence_intervals**: Whether to include confidence
      intervals for models that support it (default: true)

    - **ensemble_method**: Method for combining predictions when using
      multiple models. Options: 'mean', 'median', 'weighted' (default: 'mean')

    ### Returns:
    - Predictions from each model
    - Ensemble prediction (if multiple models used)
    - Forecast dates
    - Confidence intervals (if requested and supported)
    - Performance metrics

    ### Example Request:
    ```json
    {
        "model_name": "lstm_model.pth",
        "forecast_days": 30,
        "include_confidence_intervals": true
    }
    ```

    ### Example Request (All Models):
    ```json
    {
        "forecast_days": 7,
        "ensemble_method": "mean"
    }
    ```
    """
    try:
        # Validate forecast days
        if request.forecast_days < 1 or request.forecast_days > 365:
            raise HTTPException(
                status_code=400,
                detail="forecast_days must be between 1 and 365",
            )

        # Make predictions
        response = await prediction_service.predict(request)

        return response

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_response = PredictionErrorResponse(
            error="Prediction failed",
            detail=str(e),
            request_id="unknown",
            timestamp=datetime.now(),
        )
        raise HTTPException(status_code=500, detail=error_response.dict())


@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict_with_model(
    model_name: str, request: PredictionRequest
) -> PredictionResponse:
    """
    Make predictions using a specific model (alternative endpoint)

    ### Path Parameters:
    - **model_name**: Name of the model file
      Examples: 'lstm_model.pth', 'catboost_model', 'prophet_model.pkl'

    ### Body Parameters:
    - **forecast_days**: Number of days to forecast (1-365)
    - Other optional parameters (see /predict endpoint)

    ### Example:
    POST /predict/lstm_model.pth
    ```json
    {
        "forecast_days": 14
    }
    ```
    """
    # Override model_name in request
    request.model_name = model_name

    return await predict(request)


@app.get("/models/{model_name}/info", response_model=dict)
async def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model

    ### Parameters:
    - **model_name**: Name of the model file
    """
    try:
        available_models = prediction_service.get_available_models()

        # Find the model
        model_detail = None
        for detail in available_models.model_details:
            if detail["model_name"] == model_name:
                model_detail = detail
                break

        if model_detail is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Model '{model_name}' not found. "
                    f"Available models: {available_models.available_models}"
                ),
            )

        return model_detail

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving model info: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Global exception handler"""
    error_response = PredictionErrorResponse(
        error="Internal server error",
        detail=str(exc),
        request_id="unknown",
        timestamp=datetime.now(),
    )
    return JSONResponse(status_code=500, content=error_response.dict())


if __name__ == "__main__":
    import uvicorn

    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",  # nosec B104
        port=8001,  # Different port from training API
        reload=True,
    )
