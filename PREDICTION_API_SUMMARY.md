# API de Predicción - Resumen de Implementación

## 📦 Archivos Creados

### Componentes de la API

1. **`src/api/prediction_api.py`** (267 líneas)
   - FastAPI application principal
   - Endpoints REST para predicciones
   - Documentación Swagger automática
   - Manejo de errores robusto

2. **`src/api/prediction_service.py`** (331 líneas)
   - Servicio de predicción
   - Carga dinámica de modelos desde `src/models/training/`
   - Soporte para todos los tipos de modelos (ARIMA, Prophet, CatBoost, LightGBM, LSTM)
   - Predicción individual y ensemble

3. **`src/api/prediction_models.py`** (84 líneas)
   - Modelos Pydantic para validación
   - Request/Response schemas
   - Documentación de parámetros

### Scripts de Utilidad

4. **`start_prediction_api.py`** (23 líneas)
   - Script para iniciar el servidor fácilmente
   - Configuración lista para usar

5. **`test_prediction_api.py`** (213 líneas)
   - Suite de pruebas automatizadas
   - Verifica todos los endpoints
   - Validación de respuestas

### Documentación

6. **`PREDICTION_API_README.md`** (750+ líneas)
   - Documentación completa en español
   - Ejemplos detallados con Postman
   - Ejemplos con cURL y PowerShell
   - Solución de problemas
   - Scripts de prueba para Postman

7. **`QUICKSTART_PREDICTION_API.md`** (100+ líneas)
   - Guía rápida de inicio
   - Comandos listos para copiar/pegar
   - Troubleshooting común

8. **`postman_collection.json`**
   - Colección Postman lista para importar
   - 10 requests pre-configuradas
   - Tests automáticos incluidos

---

## 🎯 Características Implementadas

### ✅ Funcionalidad Principal

- [x] Carga dinámica de modelos desde `src/models/training/`
- [x] Predicción con modelo específico por nombre
- [x] Predicción con todos los modelos disponibles
- [x] Predicción ensemble (mean, median, weighted)
- [x] Intervalos de confianza (para modelos compatibles)
- [x] Predicción de 1 a 365 días
- [x] Validación automática de parámetros
- [x] Manejo robusto de errores

### ✅ Endpoints Implementados

1. **GET /** - Información de la API
2. **GET /health** - Health check
3. **GET /models** - Listar modelos disponibles
4. **POST /predict** - Predicción principal
5. **POST /predict/{model_name}** - Predicción con modelo en URL
6. **GET /models/{model_name}/info** - Info de modelo específico
7. **GET /docs** - Documentación Swagger (automática)
8. **GET /redoc** - Documentación ReDoc (automática)

### ✅ Modelos Soportados

- ARIMA (`.pkl`)
- SARIMAX (`.pkl`)
- Prophet (`.pkl`) ✓ Con intervalos de confianza
- CatBoost (sin extensión)
- LightGBM (`.pkl`)
- LSTM (`.pth`)

---

## 📋 Parámetros de la API

### Request Parameters

```json
{
  "model_name": "string (opcional)",      // Nombre del modelo o null para todos
  "forecast_days": "integer (1-365)",     // Días a predecir
  "data_path": "string (opcional)",       // Ruta a datos históricos
  "include_confidence_intervals": "bool", // Incluir intervalos (default: true)
  "ensemble_method": "string"             // "mean", "median", "weighted"
}
```

### Response Structure

```json
{
  "request_id": "uuid",
  "status": "success|partial|failed",
  "message": "string",
  "forecast_start_date": "YYYY-MM-DD",
  "forecast_end_date": "YYYY-MM-DD",
  "forecast_days": "integer",
  "predictions": [
    {
      "model_name": "string",
      "model_type": "string",
      "predictions": [float, ...],
      "dates": ["YYYY-MM-DD", ...],
      "confidence_lower": [float, ...] | null,
      "confidence_upper": [float, ...] | null,
      "prediction_time_seconds": float
    }
  ],
  "ensemble_prediction": [float, ...] | null,
  "ensemble_dates": ["YYYY-MM-DD", ...] | null,
  "total_models_used": integer,
  "successful_predictions": integer,
  "failed_predictions": integer,
  "total_prediction_time_seconds": float,
  "created_at": "ISO datetime"
}
```

---

## 🚀 Cómo Usar

### 1. Iniciar el Servidor

```powershell
python start_prediction_api.py
```

O:

```powershell
uvicorn src.api.prediction_api:app --reload --port 8001
```

### 2. Probar con Postman

1. Importar `postman_collection.json`
2. Ejecutar las requests

### 3. Probar con Script

```powershell
python test_prediction_api.py
```

### 4. Probar en Navegador

Abrir: http://localhost:8001/docs

---

## 💡 Ejemplos de Uso

### Ejemplo 1: Predicción con LSTM (7 días)

```powershell
$body = @{
    model_name = "lstm_model.pth"
    forecast_days = 7
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8001/predict" `
  -Method Post -ContentType "application/json" -Body $body
```

### Ejemplo 2: Predicción con Todos los Modelos (30 días)

```powershell
$body = @{
    forecast_days = 30
    ensemble_method = "mean"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8001/predict" `
  -Method Post -ContentType "application/json" -Body $body
```

### Ejemplo 3: Ver Modelos Disponibles

```powershell
Invoke-RestMethod -Uri "http://localhost:8001/models" -Method Get
```

---

## 🔧 Configuración

### Puerto del Servidor

Por defecto: `8001` (diferente de la API de training que usa `8000`)

Para cambiar:
```powershell
uvicorn src.api.prediction_api:app --reload --port NUEVO_PUERTO
```

### Directorio de Modelos

Por defecto: `src/models/training/`

Modifica en `prediction_service.py` si necesitas otro directorio.

### Datos Históricos

Por defecto: `src/preprocessing/data/raw/raw_stock_data.parquet`

Puedes especificar otro con el parámetro `data_path` en el request.

---

## ✅ Validaciones Implementadas

- ✓ Validación de formato de parámetros (Pydantic)
- ✓ Validación de rango de `forecast_days` (1-365)
- ✓ Validación de existencia de modelos
- ✓ Validación de existencia de archivos de datos
- ✓ Manejo de errores con mensajes descriptivos
- ✓ Status codes HTTP correctos

---

## 📊 Tipos de Predicción

### Individual
- Usa un modelo específico
- Retorna predicciones de ese modelo
- Incluye intervalos de confianza si aplica

### Ensemble (Múltiples Modelos)
- Usa todos los modelos disponibles
- Retorna predicciones individuales
- Retorna predicción ensemble combinada
- Métodos: mean, median, weighted

---

## 🎨 Características Adicionales

- **CORS** habilitado para desarrollo
- **Documentación automática** (Swagger + ReDoc)
- **Validación automática** de tipos
- **Logging** de requests
- **Health checks** para monitoreo
- **Request IDs** para tracking
- **Timestamps** en respuestas
- **Manejo global de excepciones**

---

## 📝 Notas de Implementación

### Decisiones de Diseño

1. **Días en lugar de fechas**: Se eligió número de días (`forecast_days`) por simplicidad
   - Más intuitivo para el usuario
   - Fácil validación (1-365)
   - Fechas se generan automáticamente

2. **Puerto 8001**: Para no interferir con la API de training (puerto 8000)

3. **Modelos desde archivo**: Carga dinámica desde `src/models/training/`
   - No requiere recompilación
   - Fácil actualización de modelos
   - Detección automática

4. **Ensemble por defecto**: Cuando no se especifica modelo
   - Aprovecha todos los modelos disponibles
   - Mejor robustez en predicciones
   - Comparación automática

### Optimizaciones Futuras Posibles

- [ ] Cache de modelos cargados
- [ ] Predicción asíncrona con task queue
- [ ] Almacenamiento de historial de predicciones
- [ ] Métricas de performance
- [ ] Rate limiting
- [ ] Autenticación/Autorización
- [ ] Soporte para batch predictions
- [ ] Exportación de resultados (CSV, Excel)

---

## 🧪 Testing

### Tests Incluidos

El archivo `test_prediction_api.py` incluye:

1. Health Check
2. Get Available Models
3. Predict with Single Model
4. Predict with All Models
5. Error Handling

### Ejecutar Tests

```powershell
python test_prediction_api.py
```

### Tests de Postman

La colección incluye tests automáticos que verifican:
- Status codes
- Estructura de respuestas
- Tipos de datos
- Valores esperados

---

## 📞 Soporte

Para más información, consultar:
- `PREDICTION_API_README.md` - Documentación completa
- `QUICKSTART_PREDICTION_API.md` - Guía rápida
- http://localhost:8001/docs - Documentación interactiva

---

**Creado**: Octubre 2025
**Versión**: 1.0.0
**Framework**: FastAPI
**Python**: 3.8+
