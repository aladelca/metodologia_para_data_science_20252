# APIs del Proyecto - Comparación

## Resumen de las Dos APIs

Este proyecto ahora cuenta con **dos APIs complementarias**:

### 1. Training API (Port 8000)
**Propósito**: Entrenar modelos de series temporales

### 2. Prediction API (Port 8001)
**Propósito**: Realizar predicciones con modelos entrenados

---

## Comparación Detallada

| Característica | Training API | Prediction API |
|---------------|--------------|----------------|
| **Puerto por defecto** | 8000 | 8001 |
| **Archivo principal** | `src/api/main.py` | `src/api/prediction_api.py` |
| **Propósito** | Entrenar modelos | Hacer predicciones |
| **Entrada principal** | Fechas de train/test | Días a predecir |
| **Salida principal** | Modelos guardados | Predicciones futuras |
| **Tiempo de ejecución** | Largo (minutos-horas) | Corto (segundos) |
| **Uso típico** | Ocasional (re-entrenar) | Frecuente (predicciones diarias) |

---

## Endpoints Principales

### Training API (Port 8000)

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Información de la API |
| `/health` | GET | Health check |
| `/train` | POST | Entrenar modelos |
| `/status/{job_id}` | GET | Estado del entrenamiento |
| `/jobs` | GET | Listar trabajos de entrenamiento |
| `/models/available` | GET | Tipos de modelos disponibles |
| `/docs` | GET | Documentación Swagger |

### Prediction API (Port 8001)

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Información de la API |
| `/health` | GET | Health check |
| `/predict` | POST | Hacer predicciones |
| `/predict/{model_name}` | POST | Predicción con modelo específico |
| `/models` | GET | Listar modelos entrenados |
| `/models/{model_name}/info` | GET | Info de modelo específico |
| `/docs` | GET | Documentación Swagger |

---

## Flujo de Trabajo Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                    FASE 1: ENTRENAMIENTO                         │
│                                                                  │
│  1. Recopilar datos históricos                                  │
│     └─> raw_stock_data.parquet                                  │
│                                                                  │
│  2. Entrenar modelos (Training API - Port 8000)                 │
│     └─> POST /train                                             │
│         {                                                        │
│           "model_types": ["arima", "lstm", "catboost"],         │
│           "train_start_date": "2021-01-01",                     │
│           "train_end_date": "2023-12-31"                        │
│         }                                                        │
│                                                                  │
│  3. Modelos guardados en src/models/training/                   │
│     └─> lstm_model.pth                                          │
│     └─> catboost_model                                          │
│     └─> prophet_model.pkl                                       │
│     └─> ...                                                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FASE 2: PREDICCIÓN                            │
│                                                                  │
│  4. Hacer predicciones (Prediction API - Port 8001)             │
│     └─> POST /predict                                           │
│         {                                                        │
│           "model_name": "lstm_model.pth",                       │
│           "forecast_days": 30                                   │
│         }                                                        │
│                                                                  │
│  5. Recibir predicciones                                        │
│     └─> predictions: [150.23, 151.45, ...]                     │
│     └─> dates: ["2024-02-01", "2024-02-02", ...]               │
│                                                                  │
│  6. Usar predicciones para toma de decisiones                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Casos de Uso

### Training API

**Cuándo usar**:
- Primera vez que se configura el sistema
- Re-entrenar modelos con nuevos datos
- Experimentar con diferentes hiperparámetros
- Agregar nuevos tipos de modelos
- Actualización periódica (ej: mensual)

**Ejemplo de uso**:
```json
POST http://localhost:8000/train
{
  "model_types": ["arima", "prophet", "lstm", "catboost"],
  "train_start_date": "2020-01-01",
  "train_end_date": "2024-12-31",
  "save_dir": "models"
}
```

### Prediction API

**Cuándo usar**:
- Predicciones diarias de mercado
- Análisis de tendencias futuras
- Comparar predicciones de múltiples modelos
- Generar reportes automáticos
- Dashboard en tiempo real

**Ejemplo de uso**:
```json
POST http://localhost:8001/predict
{
  "forecast_days": 7,
  "ensemble_method": "mean"
}
```

---

## Parámetros de Request

### Training API

```json
{
  "model_types": ["arima", "prophet", "catboost", "lightgbm", "lstm"],
  "train_start_date": "YYYY-MM-DD",
  "train_end_date": "YYYY-MM-DD",
  "test_start_date": "YYYY-MM-DD",  // Opcional
  "data_path": "string",             // Opcional
  "save_dir": "string",              // Opcional
  "arima_params": {},                // Opcional
  "prophet_params": {},              // Opcional
  "catboost_params": {},             // Opcional
  "lightgbm_params": {},             // Opcional
  "lstm_params": {}                  // Opcional
}
```

### Prediction API

```json
{
  "model_name": "lstm_model.pth",    // Opcional (null = todos)
  "forecast_days": 30,               // 1-365
  "data_path": "string",             // Opcional
  "include_confidence_intervals": true,  // Opcional
  "ensemble_method": "mean"          // "mean", "median", "weighted"
}
```

---

## Respuestas Típicas

### Training API Response

```json
{
  "job_id": "uuid",
  "status": "completed",
  "message": "Training completed successfully",
  "total_models": 5,
  "successful_models": 5,
  "failed_models": 0,
  "results": [
    {
      "model_type": "lstm",
      "status": "success",
      "model_path": "models/lstm_model.pth",
      "training_time_seconds": 45.2,
      "metrics": {...}
    }
  ],
  "total_training_time_seconds": 185.4
}
```

### Prediction API Response

```json
{
  "request_id": "uuid",
  "status": "success",
  "forecast_start_date": "2024-02-01",
  "forecast_end_date": "2024-03-01",
  "forecast_days": 30,
  "predictions": [
    {
      "model_name": "lstm_model.pth",
      "predictions": [150.23, 151.45, ...],
      "dates": ["2024-02-01", "2024-02-02", ...]
    }
  ],
  "ensemble_prediction": [150.05, 151.18, ...],
  "total_models_used": 3,
  "total_prediction_time_seconds": 0.451
}
```

---

## Cómo Ejecutar Ambas APIs

### Opción 1: Terminales Separadas

**Terminal 1 - Training API**:
```powershell
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 - Prediction API**:
```powershell
python start_prediction_api.py
# O:
uvicorn src.api.prediction_api:app --reload --port 8001
```

### Opción 2: Background Process

**PowerShell**:
```powershell
# Iniciar Training API en background
Start-Process powershell -ArgumentList "uvicorn src.api.main:app --port 8000"

# Iniciar Prediction API en foreground
python start_prediction_api.py
```

---

## Dependencias Compartidas

Ambas APIs usan:
- FastAPI
- Pydantic
- Uvicorn
- `src/pipeline/inference.py`
- `src/pipeline/training.py`
- `src/preprocessing/preprocess.py`
- Modelos en `src/models/training/`

---

## Archivos de Documentación

### Training API
- `src/api/main.py` - Código principal
- `src/api/training_service.py` - Lógica de entrenamiento
- `src/api/models.py` - Schemas

### Prediction API
- `src/api/prediction_api.py` - Código principal
- `src/api/prediction_service.py` - Lógica de predicción
- `src/api/prediction_models.py` - Schemas
- **Documentación**:
  - `PREDICTION_API_README.md` - Guía completa
  - `QUICKSTART_PREDICTION_API.md` - Inicio rápido
  - `PREDICTION_API_SUMMARY.md` - Resumen
  - `PREDICTION_API_ARCHITECTURE.md` - Arquitectura
  - `postman_collection.json` - Tests de Postman
- **Scripts**:
  - `start_prediction_api.py` - Iniciar servidor
  - `test_prediction_api.py` - Tests automáticos

---

## Workflow Recomendado

### Setup Inicial (Una vez)

1. **Entrenar modelos** (Training API):
   ```powershell
   # Iniciar Training API
   uvicorn src.api.main:app --port 8000
   
   # En Postman o script
   POST http://localhost:8000/train
   ```

2. **Verificar modelos creados**:
   ```powershell
   ls src\models\training\
   ```

### Uso Diario

1. **Iniciar Prediction API**:
   ```powershell
   python start_prediction_api.py
   ```

2. **Hacer predicciones**:
   ```powershell
   POST http://localhost:8001/predict
   ```

### Mantenimiento (Periódico)

1. **Actualizar datos**:
   - Agregar nuevos datos a `raw_stock_data.parquet`

2. **Re-entrenar modelos** (Training API):
   ```powershell
   POST http://localhost:8000/train
   ```

3. **Continuar con predicciones** (Prediction API)

---

## Tests

### Training API
- Tests en el proyecto existente
- Probar endpoints de entrenamiento

### Prediction API
```powershell
# Método 1: Script automatizado
python test_prediction_api.py

# Método 2: Postman
# Importar postman_collection.json

# Método 3: Swagger UI
# http://localhost:8001/docs
```

---

## Comparación de Performance

| Aspecto | Training API | Prediction API |
|---------|-------------|----------------|
| **Tiempo de respuesta** | Minutos-Horas | Segundos |
| **Carga de CPU** | Alta | Baja-Media |
| **Uso de memoria** | Alto | Medio |
| **Frecuencia de uso** | Ocasional | Frecuente |
| **Escalabilidad** | Batch processing | Real-time |

---

## Seguridad

Ambas APIs actualmente:
- ✓ CORS habilitado (desarrollo)
- ✓ Validación de entrada
- ✓ Manejo de errores
- ✓ Logging básico

Para producción, considerar agregar:
- [ ] Autenticación (API Keys, JWT)
- [ ] Rate limiting
- [ ] HTTPS
- [ ] Logging avanzado
- [ ] Monitoreo
- [ ] Métricas

---

## Próximos Pasos

1. **Testing**: Probar ambas APIs
2. **Documentación**: Familiarizarse con los endpoints
3. **Integración**: Conectar con sistemas externos
4. **Monitoreo**: Implementar logging y métricas
5. **Optimización**: Mejorar performance según necesidades

---

## Resumen Visual

```
┌─────────────────────────────────────────────────────────────┐
│                    ECOSISTEMA DE APIs                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │   Training API       │      │  Prediction API      │    │
│  │   Port: 8000         │      │  Port: 8001          │    │
│  ├──────────────────────┤      ├──────────────────────┤    │
│  │ • Entrena modelos    │──┬──>│ • Usa modelos        │    │
│  │ • Guarda en disco    │  │   │ • Hace predicciones  │    │
│  │ • Evalúa métricas    │  │   │ • Ensemble           │    │
│  │ • Largo tiempo       │  │   │ • Rápido             │    │
│  └──────────────────────┘  │   └──────────────────────┘    │
│                            │                                │
│                            ↓                                │
│              ┌──────────────────────────┐                   │
│              │  src/models/training/    │                   │
│              │  • lstm_model.pth        │                   │
│              │  • catboost_model        │                   │
│              │  • prophet_model.pkl     │                   │
│              └──────────────────────────┘                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

**Conclusión**: Las dos APIs trabajan en conjunto para proporcionar un sistema completo de machine learning para series temporales, separando claramente las responsabilidades de entrenamiento y predicción.
