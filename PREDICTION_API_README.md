# API de Predicción - Time Series Prediction API

API REST para realizar predicciones con modelos de series temporales entrenados (ARIMA, Prophet, CatBoost, LightGBM, LSTM).

## 📋 Tabla de Contenidos

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Endpoints](#endpoints)
- [Ejemplos con Postman](#ejemplos-con-postman)
- [Ejemplos con cURL](#ejemplos-con-curl)
- [Modelos Disponibles](#modelos-disponibles)

## ✨ Características

- Predicción con modelos individuales o todos los modelos simultáneamente
- Predicción de 1 a 365 días en el futuro
- Intervalos de confianza (para modelos compatibles)
- Predicción ensemble (promedio, mediana, o ponderada)
- API REST completamente documentada (Swagger/OpenAPI)
- Validación de parámetros automática
- Manejo robusto de errores

## 📦 Requisitos

- Python 3.8+
- FastAPI
- Uvicorn
- Todos los paquetes en `requirements.txt`

## 🚀 Instalación

1. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

2. **Verificar que existan modelos entrenados:**

Los modelos deben estar en: `src/models/training/`

Modelos soportados:
- `arima_model.pkl`
- `prophet_model.pkl`
- `catboost_model`
- `lightgbm_model.pkl`
- `lstm_model.pth`

## 🎯 Uso

### Iniciar el Servidor

**Opción 1: Ejecución directa**

```bash
cd src/api
python prediction_api.py
```

**Opción 2: Con Uvicorn**

```bash
uvicorn src.api.prediction_api:app --reload --port 8001
```

El servidor iniciará en: `http://localhost:8001`

### Acceder a la Documentación Interactiva

Una vez iniciado el servidor:

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## 📡 Endpoints

### 1. **GET /** - Información de la API

Obtiene información general de la API.

**URL**: `http://localhost:8001/`

**Respuesta**:
```json
{
  "message": "Time Series Prediction API",
  "version": "1.0.0",
  "docs": "/docs",
  "endpoints": {
    "predict": "/predict",
    "models": "/models",
    "health": "/health"
  }
}
```

---

### 2. **GET /health** - Health Check

Verifica que el servidor esté funcionando.

**URL**: `http://localhost:8001/health`

**Respuesta**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-06T10:30:00",
  "service": "time-series-prediction-api",
  "models_directory": "path/to/models"
}
```

---

### 3. **GET /models** - Listar Modelos Disponibles

Obtiene lista de todos los modelos entrenados disponibles.

**URL**: `http://localhost:8001/models`

**Respuesta**:
```json
{
  "models_directory": "src/models/training",
  "available_models": [
    "lstm_model.pth",
    "catboost_model",
    "prophet_model.pkl"
  ],
  "model_details": [
    {
      "model_name": "lstm_model.pth",
      "model_type": "lstm",
      "file_path": "src/models/training/lstm_model.pth",
      "file_size_mb": 2.45,
      "last_modified": "2025-10-05T15:30:00"
    }
  ],
  "total_models": 3
}
```

---

### 4. **POST /predict** - Realizar Predicción

Endpoint principal para realizar predicciones.

**URL**: `http://localhost:8001/predict`

**Método**: POST

**Headers**:
```
Content-Type: application/json
```

**Body (Parámetros)**:

| Parámetro | Tipo | Requerido | Default | Descripción |
|-----------|------|-----------|---------|-------------|
| `model_name` | string | No | null | Nombre del modelo específico. Si no se proporciona, usa todos |
| `forecast_days` | integer | Sí | 30 | Número de días a predecir (1-365) |
| `data_path` | string | No | default | Ruta a datos históricos |
| `include_confidence_intervals` | boolean | No | true | Incluir intervalos de confianza |
| `ensemble_method` | string | No | "mean" | Método ensemble: "mean", "median", "weighted" |

**Ejemplo 1: Predicción con un modelo específico (LSTM)**

```json
{
  "model_name": "lstm_model.pth",
  "forecast_days": 30,
  "include_confidence_intervals": true
}
```

**Ejemplo 2: Predicción con todos los modelos**

```json
{
  "forecast_days": 7,
  "ensemble_method": "mean",
  "include_confidence_intervals": true
}
```

**Ejemplo 3: Predicción rápida de 14 días con CatBoost**

```json
{
  "model_name": "catboost_model",
  "forecast_days": 14,
  "include_confidence_intervals": false
}
```

**Respuesta de Éxito (200)**:

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "message": "Successfully generated predictions for 2 model(s)",
  "forecast_start_date": "2024-02-01",
  "forecast_end_date": "2024-03-01",
  "forecast_days": 30,
  "predictions": [
    {
      "model_name": "lstm_model.pth",
      "model_type": "lstm",
      "predictions": [150.23, 151.45, 152.67, ...],
      "dates": ["2024-02-01", "2024-02-02", "2024-02-03", ...],
      "confidence_lower": null,
      "confidence_upper": null,
      "prediction_time_seconds": 0.234
    },
    {
      "model_name": "catboost_model",
      "model_type": "catboost",
      "predictions": [149.88, 150.92, 151.98, ...],
      "dates": ["2024-02-01", "2024-02-02", "2024-02-03", ...],
      "confidence_lower": null,
      "confidence_upper": null,
      "prediction_time_seconds": 0.145
    }
  ],
  "ensemble_prediction": [150.05, 151.18, 152.32, ...],
  "ensemble_dates": ["2024-02-01", "2024-02-02", "2024-02-03", ...],
  "total_models_used": 2,
  "successful_predictions": 2,
  "failed_predictions": 0,
  "total_prediction_time_seconds": 0.451,
  "created_at": "2025-10-06T10:30:00"
}
```

---

### 5. **POST /predict/{model_name}** - Predicción con Modelo Específico

Endpoint alternativo para especificar el modelo en la URL.

**URL**: `http://localhost:8001/predict/lstm_model.pth`

**Método**: POST

**Body**:
```json
{
  "forecast_days": 14
}
```

---

### 6. **GET /models/{model_name}/info** - Información del Modelo

Obtiene información detallada de un modelo específico.

**URL**: `http://localhost:8001/models/lstm_model.pth/info`

**Respuesta**:
```json
{
  "model_name": "lstm_model.pth",
  "model_type": "lstm",
  "file_path": "src/models/training/lstm_model.pth",
  "file_size_mb": 2.45,
  "last_modified": "2025-10-05T15:30:00"
}
```

---

## 🧪 Ejemplos con Postman

### Configuración Inicial en Postman

1. **Crear nueva Collection**: "Time Series Prediction API"
2. **Configurar variables de entorno**:
   - `base_url`: `http://localhost:8001`

### Test 1: Verificar que la API está funcionando

**Request**: GET Health Check

```
GET {{base_url}}/health
```

**Tests Script** (en Postman):
```javascript
pm.test("Status is healthy", function () {
    pm.response.to.have.status(200);
    var jsonData = pm.response.json();
    pm.expect(jsonData.status).to.eql("healthy");
});
```

---

### Test 2: Listar Modelos Disponibles

**Request**: GET Models

```
GET {{base_url}}/models
```

**Tests Script**:
```javascript
pm.test("Models found", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.total_models).to.be.above(0);
    pm.expect(jsonData.available_models).to.be.an('array');
});

// Guardar primer modelo para siguientes tests
var models = pm.response.json().available_models;
if (models.length > 0) {
    pm.environment.set("first_model", models[0]);
}
```

---

### Test 3: Predicción con Modelo Específico (LSTM)

**Request**: POST Predict LSTM

```
POST {{base_url}}/predict
Content-Type: application/json

Body:
{
  "model_name": "lstm_model.pth",
  "forecast_days": 7,
  "include_confidence_intervals": false
}
```

**Tests Script**:
```javascript
pm.test("Prediction successful", function () {
    pm.response.to.have.status(200);
    var jsonData = pm.response.json();
    pm.expect(jsonData.status).to.eql("success");
    pm.expect(jsonData.predictions).to.be.an('array').that.is.not.empty;
});

pm.test("Correct forecast days", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.forecast_days).to.eql(7);
});

pm.test("Has predictions", function () {
    var jsonData = pm.response.json();
    var firstPred = jsonData.predictions[0];
    pm.expect(firstPred.predictions).to.be.an('array');
    pm.expect(firstPred.dates).to.be.an('array');
});
```

---

### Test 4: Predicción con Todos los Modelos

**Request**: POST Predict All Models

```
POST {{base_url}}/predict
Content-Type: application/json

Body:
{
  "forecast_days": 30,
  "ensemble_method": "mean",
  "include_confidence_intervals": true
}
```

**Tests Script**:
```javascript
pm.test("Multiple models used", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.total_models_used).to.be.above(1);
});

pm.test("Ensemble prediction exists", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.ensemble_prediction).to.exist;
    pm.expect(jsonData.ensemble_dates).to.exist;
});
```

---

### Test 5: Predicción con Parámetro en URL

**Request**: POST Predict with URL Param

```
POST {{base_url}}/predict/catboost_model
Content-Type: application/json

Body:
{
  "forecast_days": 14
}
```

---

### Test 6: Error Handling - Modelo No Existe

**Request**: POST Invalid Model

```
POST {{base_url}}/predict
Content-Type: application/json

Body:
{
  "model_name": "modelo_inexistente.pkl",
  "forecast_days": 7
}
```

**Tests Script**:
```javascript
pm.test("Should return 400 error", function () {
    pm.response.to.have.status(400);
});
```

---

## 💻 Ejemplos con cURL

### Ejemplo 1: Health Check

```bash
curl -X GET http://localhost:8001/health
```

### Ejemplo 2: Listar Modelos

```bash
curl -X GET http://localhost:8001/models
```

### Ejemplo 3: Predicción con LSTM (7 días)

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lstm_model.pth",
    "forecast_days": 7,
    "include_confidence_intervals": false
  }'
```

### Ejemplo 4: Predicción con Todos los Modelos (30 días)

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "forecast_days": 30,
    "ensemble_method": "mean"
  }'
```

### Ejemplo 5: Predicción con CatBoost (formato compacto)

```bash
curl -X POST http://localhost:8001/predict/catboost_model \
  -H "Content-Type: application/json" \
  -d '{"forecast_days": 14}'
```

### Ejemplo 6: Windows PowerShell

```powershell
# Health Check
Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get

# Listar modelos
Invoke-RestMethod -Uri "http://localhost:8001/models" -Method Get

# Predicción
$body = @{
    model_name = "lstm_model.pth"
    forecast_days = 7
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8001/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

---

## 📊 Modelos Disponibles

| Modelo | Archivo | Tipo | Intervalos de Confianza | Descripción |
|--------|---------|------|------------------------|-------------|
| ARIMA | `arima_model.pkl` | Estadístico | ✅ Sí | Auto-regresivo integrado de media móvil |
| SARIMAX | `sarimax_model.pkl` | Estadístico | ✅ Sí | ARIMA con componentes estacionales |
| Prophet | `prophet_model.pkl` | Estadístico | ✅ Sí | Modelo de Facebook para series temporales |
| CatBoost | `catboost_model` | Machine Learning | ❌ No | Gradient Boosting (CatBoost) |
| LightGBM | `lightgbm_model.pkl` | Machine Learning | ❌ No | Gradient Boosting (LightGBM) |
| LSTM | `lstm_model.pth` | Deep Learning | ❌ No | Red neuronal LSTM |

---

## 🔧 Solución de Problemas

### Error: "No trained models found"

**Solución**: Asegúrate de que existan modelos entrenados en `src/models/training/`

```bash
# Verificar modelos
ls src/models/training/
```

### Error: "Data file not found"

**Solución**: Proporciona la ruta correcta al archivo de datos o asegúrate de que existe el archivo por defecto.

```json
{
  "data_path": "src/preprocessing/data/raw/raw_stock_data.parquet",
  "forecast_days": 30
}
```

### Puerto en uso

**Solución**: Cambia el puerto al iniciar el servidor:

```bash
uvicorn src.api.prediction_api:app --reload --port 8002
```

---

## 📝 Notas Adicionales

- Los modelos ML (CatBoost, LightGBM) requieren datos de prueba preparados
- LSTM usa escaladores guardados durante el entrenamiento
- La predicción ensemble solo está disponible cuando se usan múltiples modelos
- Los intervalos de confianza solo están disponibles para modelos estadísticos (ARIMA, Prophet)

---

## 🤝 Contribuciones

Para reportar problemas o sugerir mejoras, contacta al equipo de desarrollo.

---

## 📄 Licencia

Este proyecto es parte del curso de Metodología para Data Science 2025-2.
