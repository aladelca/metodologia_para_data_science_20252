"""
Prediction service for handling model inference operations
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from pipeline.inference import TimeSeriesInference
from preprocessing.preprocess import TimeSeriesPreprocessor

from .models import (
    ModelType,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    ModelInfoResponse,
)


class PredictionService:
    """Service class for handling prediction operations"""

    def __init__(self):
        self.preprocessor = TimeSeriesPreprocessor()
        self.inference = TimeSeriesInference(self.preprocessor)
        self.active_predictions: Dict[str, PredictionResponse] = {}

    async def predict(
        self, request: PredictionRequest
    ) -> PredictionResponse:
        """
        Make predictions using trained models

        Args:
            request: Prediction request with model info and parameters

        Returns:
            PredictionResponse with forecast results
        """
        prediction_id = str(uuid.uuid4())

        try:
            # Validate model file exists
            if not os.path.exists(request.model_path):
                raise FileNotFoundError(
                    f"Model file not found: {request.model_path}"
                )

            # Make predictions based on model type
            predictions = await self._make_predictions(request)

            response = PredictionResponse(
                prediction_id=prediction_id,
                status="success",
                message=f"Prediction completed for {request.model_type}",
                model_type=request.model_type,
                model_path=request.model_path,
                periods=request.periods,
                predictions=predictions,
                created_at=datetime.now()
            )

            self.active_predictions[prediction_id] = response
            return response

        except Exception as e:
            error_response = PredictionResponse(
                prediction_id=prediction_id,
                status="failed",
                message=f"Prediction failed: {str(e)}",
                model_type=request.model_type,
                model_path=request.model_path,
                periods=request.periods,
                predictions=[],
                created_at=datetime.now()
            )
            self.active_predictions[prediction_id] = error_response
            return error_response

    async def _make_predictions(
        self, request: PredictionRequest
    ) -> List[PredictionResult]:
        """Make predictions for a specific model type"""
        
        try:
            # Load the model
            model = self.inference.load_model(
                request.model_type.value, 
                request.model_path
            )

            if request.model_type == ModelType.ARIMA:
                return await self._predict_arima(model, request)
            elif request.model_type == ModelType.PROPHET:
                return await self._predict_prophet(model, request)
            elif request.model_type == ModelType.CATBOOST:
                return await self._predict_catboost(model, request)
            elif request.model_type == ModelType.LIGHTGBM:
                return await self._predict_lightgbm(model, request)
            elif request.model_type == ModelType.LSTM:
                return await self._predict_lstm(model, request)
            else:
                raise ValueError(f"Unsupported model type: {request.model_type}")

        except Exception as e:
            raise Exception(f"Prediction failed for {request.model_type}: {str(e)}")

    async def _predict_arima(
        self, model, request: PredictionRequest
    ) -> List[PredictionResult]:
        """Make ARIMA predictions - NO necesita data_path"""
        # Make forecast
        forecast, conf_int = self.inference.predict_arima(
            model, 
            steps=request.periods, 
            return_conf_int=True
        )

        predictions = []
        start_date = datetime.now()
        
        for i in range(request.periods):
            date = start_date + timedelta(days=i)
            predictions.append(
                PredictionResult(
                    timestamp=date.isoformat(),
                    date=date.date().isoformat(),
                    value=float(forecast.iloc[i]),
                    confidence_lower=float(conf_int.iloc[i, 0]) if conf_int is not None else None,
                    confidence_upper=float(conf_int.iloc[i, 1]) if conf_int is not None else None,
                    model_type=ModelType.ARIMA.value
                )
            )
        
        return predictions

    async def _predict_prophet(
        self, model, request: PredictionRequest
    ) -> List[PredictionResult]:
        """Make Prophet predictions - NO necesita data_path"""
        # Make forecast
        forecast_df = self.inference.predict_prophet(
            model, 
            periods=request.periods,
            freq=request.prediction_params.get("freq", "D")
        )

        predictions = []
        for _, row in forecast_df.tail(request.periods).iterrows():
            predictions.append(
                PredictionResult(
                    timestamp=row['ds'].isoformat(),
                    date=row['ds'].date().isoformat(),
                    value=float(row['yhat']),
                    confidence_lower=float(row.get('yhat_lower', row['yhat'])),
                    confidence_upper=float(row.get('yhat_upper', row['yhat'])),
                    model_type=ModelType.PROPHET.value
                )
            )
        
        return predictions

    async def _predict_catboost(
        self, model, request: PredictionRequest
    ) -> List[PredictionResult]:
        """Make CatBoost predictions - funciona CON o SIN data_path"""
        try:
            if request.data_path and os.path.exists(request.data_path):
                # Método 1: Con datos - para evaluación
                data = self.preprocessor.load_data(request.data_path)
                _, _, test_X, _ = self.preprocessor.prepare_ml_data(data)
                
                predictions_array = self.inference.predict_ml_models(
                    model, test_X, "catboost"
                )
                
                # Usar solo el número de predicciones solicitado
                predictions_to_use = min(request.periods, len(predictions_array))
                predictions_array = predictions_array[:predictions_to_use]
                
            else:
                # Método 2: Sin datos - crear features dummy para forecasting
                # Esto es una simplificación - en producción necesitarías una mejor lógica
                n_features = 10  # Número estimado de features
                dummy_features = np.random.normal(0, 1, (request.periods, n_features))
                
                predictions_array = self.inference.predict_ml_models(
                    model, dummy_features, "catboost"
                )
            
            predictions = []
            start_date = datetime.now()
            
            for i in range(len(predictions_array)):
                date = start_date + timedelta(days=i)
                predictions.append(
                    PredictionResult(
                        timestamp=date.isoformat(),
                        date=date.date().isoformat(),
                        value=float(predictions_array[i]),
                        model_type=ModelType.CATBOOST.value
                    )
                )
            
            return predictions
            
        except Exception as e:
            raise Exception(f"CatBoost prediction error: {str(e)}")

    async def _predict_lightgbm(
        self, model, request: PredictionRequest
    ) -> List[PredictionResult]:
        """Make LightGBM predictions - funciona CON o SIN data_path"""
        try:
            if request.data_path and os.path.exists(request.data_path):
                # Método 1: Con datos - para evaluación
                data = self.preprocessor.load_data(request.data_path)
                _, _, test_X, _ = self.preprocessor.prepare_ml_data(data)
                
                predictions_array = self.inference.predict_ml_models(
                    model, test_X, "lightgbm"
                )
                
                # Usar solo el número de predicciones solicitado
                predictions_to_use = min(request.periods, len(predictions_array))
                predictions_array = predictions_array[:predictions_to_use]
                
            else:
                # Método 2: Sin datos - crear features dummy para forecasting
                n_features = 10  # Número estimado de features
                dummy_features = np.random.normal(0, 1, (request.periods, n_features))
                
                predictions_array = self.inference.predict_ml_models(
                    model, dummy_features, "lightgbm"
                )
            
            predictions = []
            start_date = datetime.now()
            
            for i in range(len(predictions_array)):
                date = start_date + timedelta(days=i)
                predictions.append(
                    PredictionResult(
                        timestamp=date.isoformat(),
                        date=date.date().isoformat(),
                        value=float(predictions_array[i]),
                        model_type=ModelType.LIGHTGBM.value
                    )
                )
            
            return predictions
            
        except Exception as e:
            raise Exception(f"LightGBM prediction error: {str(e)}")

    async def _predict_lstm(
        self, model, request: PredictionRequest
    ) -> List[PredictionResult]:
        """Make LSTM predictions - funciona CON o SIN data_path"""
        try:
            # Obtener parámetros
            sequence_length = request.prediction_params.get("sequence_length", 30)
            target_scaler = self.preprocessor.target_scaler
            
            if request.data_path and os.path.exists(request.data_path):
                # Método 1: Con datos - para evaluación
                data = self.preprocessor.load_data(request.data_path)
                _, X_test, _, _ = self.preprocessor.prepare_lstm_data(data)
                
                predictions_array = self.inference.predict_lstm(
                    model, X_test, target_scaler
                )
                
                # Usar solo el número de predicciones solicitado
                predictions_to_use = min(request.periods, len(predictions_array))
                predictions_array = predictions_array[:predictions_to_use]
                
            else:
                # Método 2: Sin datos - forecasting real multi-step
                # Crear una secuencia inicial (en producción esto debería venir del entrenamiento)
                initial_sequence = np.random.normal(0, 1, (sequence_length, 1))
                
                predictions_array = self.inference.multi_step_forecast_lstm(
                    model, 
                    initial_sequence, 
                    steps=request.periods,
                    target_scaler=target_scaler
                )
            
            predictions = []
            start_date = datetime.now()
            
            for i in range(len(predictions_array)):
                date = start_date + timedelta(days=i)
                predictions.append(
                    PredictionResult(
                        timestamp=date.isoformat(),
                        date=date.date().isoformat(),
                        value=float(predictions_array[i]),
                        model_type=ModelType.LSTM.value
                    )
                )
            
            return predictions
            
        except Exception as e:
            raise Exception(f"LSTM prediction error: {str(e)}")

    async def predict_all_models(
        self, 
        models_dir: str, 
        data_path: Optional[str] = None,  # ← Ahora es opcional
        periods: int = 30
    ) -> Dict[str, List[PredictionResult]]:
        """
        Make predictions using all available trained models
        
        Args:
            models_dir: Directory containing trained models
            data_path: Optional path to data file (para evaluación)
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with predictions for each model type
        """
        all_predictions = {}

        # Test each model type
        model_types = [
            (ModelType.ARIMA, "arima_model.pkl"),
            (ModelType.PROPHET, "prophet_model.pkl"),
            (ModelType.CATBOOST, "catboost_model"),
            (ModelType.LIGHTGBM, "lightgbm_model.pkl"),
            (ModelType.LSTM, "lstm_model.pth")
        ]

        for model_type, model_file in model_types:
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                try:
                    request = PredictionRequest(
                        model_type=model_type,
                        model_path=model_path,
                        data_path=data_path,  # Puede ser None
                        periods=periods
                    )
                    predictions = await self._make_predictions(request)
                    all_predictions[model_type.value] = predictions
                except Exception as e:
                    print(f"Error predicting with {model_type}: {e}")

        return all_predictions

    def get_prediction_status(self, prediction_id: str) -> Optional[PredictionResponse]:
        """Get the status of a prediction job"""
        return self.active_predictions.get(prediction_id)

    def list_predictions(self) -> List[PredictionResponse]:
        """List all prediction jobs"""
        return list(self.active_predictions.values())