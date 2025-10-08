# 📚 Índice de Documentación - API de Predicción

## 🚀 Inicio Rápido

**¿Primera vez?** Empieza aquí:

1. **[QUICKSTART_PREDICTION_API.md](QUICKSTART_PREDICTION_API.md)** ⭐
   - Guía de inicio en 3 pasos
   - Comandos listos para usar
   - ~5 minutos para empezar

2. **Iniciar el servidor**:
   ```powershell
   python start_prediction_api.py
   ```

3. **Abrir en navegador**:
   - http://localhost:8001/docs

---

## 📖 Documentación Completa

### Guías de Usuario

| Documento | Descripción | Cuándo Leer |
|-----------|-------------|-------------|
| **[QUICKSTART_PREDICTION_API.md](QUICKSTART_PREDICTION_API.md)** | Guía rápida de inicio | 🟢 **Empezar aquí** |
| **[PREDICTION_API_README.md](PREDICTION_API_README.md)** | Documentación completa con ejemplos | Después del quickstart |
| **[PREDICTION_API_COMPLETED.md](PREDICTION_API_COMPLETED.md)** | Resumen de lo implementado | Para overview rápido |

### Guías Técnicas

| Documento | Descripción | Audiencia |
|-----------|-------------|-----------|
| **[PREDICTION_API_SUMMARY.md](PREDICTION_API_SUMMARY.md)** | Resumen técnico e implementación | Desarrolladores |
| **[PREDICTION_API_ARCHITECTURE.md](PREDICTION_API_ARCHITECTURE.md)** | Arquitectura y diagramas | Arquitectos/Devs |
| **[APIs_COMPARISON.md](APIs_COMPARISON.md)** | Training vs Prediction API | Todos |

---

## 🛠️ Herramientas

### Scripts

| Script | Propósito | Comando |
|--------|-----------|---------|
| **[start_prediction_api.py](start_prediction_api.py)** | Iniciar servidor | `python start_prediction_api.py` |
| **[test_prediction_api.py](test_prediction_api.py)** | Tests automáticos | `python test_prediction_api.py` |

### Testing

| Herramienta | Archivo | Uso |
|-------------|---------|-----|
| **Postman Collection** | [postman_collection.json](postman_collection.json) | Importar en Postman |
| **Swagger UI** | - | http://localhost:8001/docs |
| **ReDoc** | - | http://localhost:8001/redoc |

---

## 🎯 Por Caso de Uso

### Quiero hacer una predicción rápida
1. Leer: [QUICKSTART_PREDICTION_API.md](QUICKSTART_PREDICTION_API.md)
2. Ejecutar: `python start_prediction_api.py`
3. Abrir: http://localhost:8001/docs
4. Usar endpoint: `POST /predict`

### Quiero entender cómo funciona
1. Leer: [PREDICTION_API_ARCHITECTURE.md](PREDICTION_API_ARCHITECTURE.md)
2. Revisar código: `src/api/prediction_api.py`

### Quiero probarlo con Postman
1. Importar: [postman_collection.json](postman_collection.json)
2. Configurar variable: `base_url = http://localhost:8001`
3. Ejecutar requests

### Quiero comparar con Training API
1. Leer: [APIs_COMPARISON.md](APIs_COMPARISON.md)

### Quiero ver todos los detalles
1. Leer: [PREDICTION_API_README.md](PREDICTION_API_README.md)

---

## 📂 Estructura de Archivos

```
metodologia_para_data_science_20252/
│
├── 📚 DOCUMENTACIÓN
│   ├── QUICKSTART_PREDICTION_API.md ⭐ (Empezar aquí)
│   ├── PREDICTION_API_README.md (Guía completa)
│   ├── PREDICTION_API_COMPLETED.md (Resumen)
│   ├── PREDICTION_API_SUMMARY.md (Técnico)
│   ├── PREDICTION_API_ARCHITECTURE.md (Arquitectura)
│   └── APIs_COMPARISON.md (Comparación)
│
├── 🚀 SCRIPTS
│   ├── start_prediction_api.py (Iniciar servidor)
│   └── test_prediction_api.py (Tests)
│
├── 🧪 TESTING
│   └── postman_collection.json (Postman)
│
└── 💻 CÓDIGO
    └── src/api/
        ├── prediction_api.py (API endpoints)
        ├── prediction_service.py (Lógica)
        └── prediction_models.py (Schemas)
```

---

## ⚡ Comandos Rápidos

### PowerShell

```powershell
# 1. Activar entorno virtual
.venv\Scripts\Activate

# 2. Instalar dependencias (si es necesario)
pip install -r requirements.txt

# 3. Iniciar servidor
python start_prediction_api.py

# 4. En otra terminal - Ejecutar tests
python test_prediction_api.py

# 5. Ver modelos disponibles
Invoke-RestMethod -Uri "http://localhost:8001/models" -Method Get

# 6. Hacer predicción con LSTM (7 días)
$body = @{ model_name = "lstm_model.pth"; forecast_days = 7 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method Post -ContentType "application/json" -Body $body
```

---

## 🎓 Flujo de Aprendizaje Recomendado

### Nivel 1: Básico (15 minutos)
1. ✅ Leer [QUICKSTART_PREDICTION_API.md](QUICKSTART_PREDICTION_API.md)
2. ✅ Ejecutar `python start_prediction_api.py`
3. ✅ Abrir http://localhost:8001/docs
4. ✅ Probar endpoint `/models` (GET)
5. ✅ Probar endpoint `/predict` (POST) con Swagger UI

### Nivel 2: Intermedio (30 minutos)
1. ✅ Leer [PREDICTION_API_README.md](PREDICTION_API_README.md)
2. ✅ Importar [postman_collection.json](postman_collection.json)
3. ✅ Ejecutar todas las requests de Postman
4. ✅ Ejecutar `python test_prediction_api.py`
5. ✅ Experimentar con diferentes parámetros

### Nivel 3: Avanzado (1 hora)
1. ✅ Leer [PREDICTION_API_ARCHITECTURE.md](PREDICTION_API_ARCHITECTURE.md)
2. ✅ Revisar código de `src/api/prediction_api.py`
3. ✅ Revisar código de `src/api/prediction_service.py`
4. ✅ Leer [APIs_COMPARISON.md](APIs_COMPARISON.md)
5. ✅ Personalizar y extender la API

---

## 📊 Endpoints Principales

| Endpoint | Método | Descripción | Documentación |
|----------|--------|-------------|---------------|
| `/health` | GET | Health check | Todas las guías |
| `/models` | GET | Listar modelos | [README](PREDICTION_API_README.md#3-get-models---listar-modelos-disponibles) |
| `/predict` | POST | Hacer predicción | [README](PREDICTION_API_README.md#4-post-predict---realizar-predicción) |
| `/docs` | GET | Swagger UI | Auto-generada |

---

## 🆘 Ayuda y Soporte

### Problema Común 1: No hay modelos
**Síntoma**: "No trained models found"
**Solución**: [PREDICTION_API_README.md - Troubleshooting](PREDICTION_API_README.md#error-no-trained-models-found)

### Problema Común 2: Puerto en uso
**Síntoma**: "Address already in use"
**Solución**: [QUICKSTART_PREDICTION_API.md - Troubleshooting](QUICKSTART_PREDICTION_API.md#error-puerto-en-uso)

### Problema Común 3: Módulos no encontrados
**Síntoma**: "ModuleNotFoundError"
**Solución**: 
```powershell
pip install -r requirements.txt
```

---

## 🔗 Enlaces Rápidos

- **Servidor Local**: http://localhost:8001
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **Health Check**: http://localhost:8001/health

---

## ✅ Checklist de Verificación

Antes de usar la API, verifica:

- [ ] Python 3.8+ instalado
- [ ] Entorno virtual activado (`.venv`)
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Modelos entrenados existen en `src/models/training/`
- [ ] Servidor iniciado (`python start_prediction_api.py`)
- [ ] API responde en http://localhost:8001/health

---

## 📞 Información de Contacto

**Versión**: 1.0.0
**Framework**: FastAPI
**Puerto**: 8001
**Fecha**: Octubre 2025

---

## 🎯 Próximos Pasos

1. **[Leer Quickstart](QUICKSTART_PREDICTION_API.md)**
2. **Iniciar servidor**: `python start_prediction_api.py`
3. **Explorar Swagger**: http://localhost:8001/docs
4. **Hacer primera predicción**
5. **Leer documentación completa**: [PREDICTION_API_README.md](PREDICTION_API_README.md)

---

**¡Comienza ahora con [QUICKSTART_PREDICTION_API.md](QUICKSTART_PREDICTION_API.md)!** 🚀
