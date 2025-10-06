# Guía Rápida - API de Predicción

## 🚀 Inicio Rápido en 3 Pasos

### 1. Instalar Dependencias

```powershell
# Activar entorno virtual (si no está activado)
.venv\Scripts\Activate

# Instalar/actualizar dependencias
pip install -r requirements.txt
```

### 2. Iniciar el Servidor

```powershell
# Método 1: Script de inicio (Recomendado)
python start_prediction_api.py

# Método 2: Uvicorn directo
uvicorn src.api.prediction_api:app --reload --port 8001
```

El servidor estará disponible en: **http://localhost:8001**

### 3. Probar la API

**Opción A: Navegador**
- Abrir: http://localhost:8001/docs
- Usar la interfaz interactiva de Swagger

**Opción B: Script de Prueba**
```powershell
# En otra terminal (mantener el servidor corriendo)
python test_prediction_api.py
```

**Opción C: Postman**
1. Importar `postman_collection.json`
2. Ejecutar las pruebas

---

## 📋 Ejemplos Rápidos

### PowerShell

```powershell
# 1. Health Check
Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get

# 2. Ver modelos disponibles
Invoke-RestMethod -Uri "http://localhost:8001/models" -Method Get

# 3. Predicción con LSTM (7 días)
$body = @{
    model_name = "lstm_model.pth"
    forecast_days = 7
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8001/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

# 4. Predicción con todos los modelos (30 días)
$body = @{
    forecast_days = 30
    ensemble_method = "mean"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8001/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

---

## 📁 Archivos Importantes

| Archivo | Descripción |
|---------|-------------|
| `start_prediction_api.py` | Script para iniciar el servidor |
| `test_prediction_api.py` | Script de pruebas automáticas |
| `PREDICTION_API_README.md` | Documentación completa |
| `postman_collection.json` | Colección de Postman |
| `src/api/prediction_api.py` | Endpoints de la API |
| `src/api/prediction_service.py` | Lógica de predicción |
| `src/api/prediction_models.py` | Modelos Pydantic |

---

## 🎯 Endpoints Principales

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Verificar estado del servidor |
| `/models` | GET | Listar modelos disponibles |
| `/predict` | POST | Hacer predicción |
| `/docs` | GET | Documentación Swagger |

---

## ⚙️ Parámetros de Predicción

```json
{
  "model_name": "lstm_model.pth",  // Opcional: si no se especifica, usa todos
  "forecast_days": 30,              // Requerido: días a predecir (1-365)
  "include_confidence_intervals": true,  // Opcional: intervalos de confianza
  "ensemble_method": "mean"         // Opcional: "mean", "median", "weighted"
}
```

---

## 🔧 Solución de Problemas

### Error: "No trained models found"
```powershell
# Verificar que existen modelos
ls src\models\training\
```

### Error: Puerto en uso
```powershell
# Usar otro puerto
uvicorn src.api.prediction_api:app --reload --port 8002
```

### Error: ModuleNotFoundError
```powershell
# Reinstalar dependencias
pip install -r requirements.txt
```

---

## 📊 Modelos Disponibles

- **lstm_model.pth** - Red neuronal LSTM
- **catboost_model** - Gradient Boosting (CatBoost)
- **prophet_model.pkl** - Prophet (Facebook)
- **arima_model.pkl** - ARIMA
- **lightgbm_model.pkl** - LightGBM

---

## 📖 Documentación Completa

Para más detalles, consultar: `PREDICTION_API_README.md`
