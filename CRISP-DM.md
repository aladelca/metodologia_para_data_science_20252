# Metodología CRISP-DM para Data Science

## Introducción

CRISP-DM (Cross-Industry Standard Process for Data Mining) es una metodología robusta y ampliamente adoptada para proyectos de data science y minería de datos. Proporciona un marco estructurado que guía a los equipos a través de todo el ciclo de vida de un proyecto de datos, desde la comprensión inicial del problema de negocio hasta la implementación y monitoreo de la solución.

## Visión General de CRISP-DM

CRISP-DM consta de **6 fases principales** que se ejecutan de manera iterativa y cíclica:

1. **Comprensión del Negocio** (Business Understanding)
2. **Comprensión de los Datos** (Data Understanding)
3. **Preparación de los Datos** (Data Preparation)
4. **Modelado** (Modeling)
5. **Evaluación** (Evaluation)
6. **Implementación** (Deployment)

### Características Clave

- **Iterativo**: Las fases pueden repetirse según sea necesario
- **Flexible**: Se adapta a diferentes tipos de proyectos y industrias
- **Orientado al negocio**: Comienza y termina con consideraciones empresariales
- **Basado en evidencia**: Cada decisión se basa en análisis y validación

## Fase 1: Comprensión del Negocio

### Objetivo
Entender los objetivos del proyecto desde una perspectiva empresarial y convertir este conocimiento en una definición del problema de data science.

### Actividades Principales

#### 1.1 Determinar Objetivos del Negocio
- **Antecedentes**: Contexto y situación actual del negocio
- **Objetivos empresariales**: Qué quiere lograr la organización
- **Criterios de éxito**: Cómo se medirá el éxito del proyecto

**Ejemplo práctico:**
```
Antecedentes: Una empresa de retail quiere reducir la pérdida de clientes
Objetivo: Reducir la tasa de abandono de clientes en un 20% en 6 meses
Criterios de éxito: Aumento en la retención de clientes, incremento en LTV
```

#### 1.2 Evaluar la Situación
- **Inventario de recursos**: Datos, personal, tecnología disponible
- **Requisitos y restricciones**: Limitaciones técnicas, legales, presupuestarias
- **Riesgos y contingencias**: Identificación de posibles obstáculos

#### 1.3 Determinar Objetivos de Data Science
- **Objetivos de data science**: Traducción de objetivos empresariales a técnicos
- **Criterios de éxito técnicos**: Métricas y umbrales específicos

**Ejemplo de traducción:**
```
Objetivo empresarial: Reducir abandono de clientes
Objetivo de data science: Desarrollar un modelo predictivo que identifique 
clientes con alta probabilidad de abandono (precisión > 85%)
```

#### 1.4 Producir Plan del Proyecto
- **Cronograma**: Fases, hitos y entregas
- **Evaluación inicial de herramientas**: Tecnologías a utilizar
- **Análisis costo-beneficio**: ROI esperado del proyecto

### Entregables
- Documento de objetivos del negocio
- Plan del proyecto
- Evaluación de la situación
- Definición de objetivos de data science

## Fase 2: Comprensión de los Datos

### Objetivo
Familiarizarse con los datos disponibles e identificar problemas de calidad, patrones interesantes y subconjuntos de datos relevantes.

### Actividades Principales

#### 2.1 Recopilar Datos Iniciales
- **Adquisición de datos**: Obtener acceso a las fuentes de datos
- **Carga de datos**: Importar datos al entorno de trabajo
- **Verificación de carga**: Confirmar integridad en la transferencia

#### 2.2 Describir los Datos
- **Análisis descriptivo básico**: Cantidad, formato, identidad de atributos
- **Exploración de la estructura**: Relaciones entre tablas/archivos
- **Documentación de metadatos**: Significado y origen de cada variable

**Ejemplo de descripción:**
```python
# Ejemplo de análisis descriptivo
import pandas as pd

df = pd.read_csv('customer_data.csv')
print(f"Dimensiones: {df.shape}")
print(f"Tipos de datos:\n{df.dtypes}")
print(f"Valores nulos:\n{df.isnull().sum()}")
print(f"Estadísticas descriptivas:\n{df.describe()}")
```

#### 2.3 Explorar los Datos
- **Análisis exploratorio de datos (EDA)**: Distribuciones, correlaciones, outliers
- **Visualizaciones**: Gráficos que revelen patrones y anomalías
- **Segmentación inicial**: Identificación de subgrupos relevantes

#### 2.4 Verificar la Calidad de los Datos
- **Problemas de calidad**: Valores faltantes, duplicados, inconsistencias
- **Evaluación de completitud**: Porcentaje de datos disponibles
- **Análisis de outliers**: Identificación de valores atípicos

### Entregables
- Reporte de recopilación de datos
- Reporte de descripción de datos
- Reporte de exploración de datos
- Reporte de calidad de datos

## Fase 3: Preparación de los Datos

### Objetivo
Construir el conjunto de datos final que será alimentado a las herramientas de modelado.

### Actividades Principales

#### 3.1 Seleccionar Datos
- **Criterios de inclusión/exclusión**: Qué datos usar y cuáles descartar
- **Justificación técnica**: Razones para la selección
- **Impacto en los objetivos**: Cómo la selección afecta los resultados

#### 3.2 Limpiar Datos
- **Tratamiento de valores faltantes**: Imputación, eliminación, marcado
- **Corrección de inconsistencias**: Estandarización de formatos
- **Eliminación de duplicados**: Identificación y manejo de registros repetidos

**Ejemplo de limpieza:**
```python
# Tratamiento de valores faltantes
df['age'].fillna(df['age'].median(), inplace=True)

# Estandarización de formatos
df['email'] = df['email'].str.lower().str.strip()

# Eliminación de duplicados
df = df.drop_duplicates(subset=['customer_id'])
```

#### 3.3 Construir Datos
- **Feature engineering**: Creación de nuevas variables derivadas
- **Agregaciones**: Resúmenes y métricas calculadas
- **Transformaciones**: Normalización, escalado, codificación

#### 3.4 Integrar Datos
- **Combinación de fuentes**: Joins, merges, concatenaciones
- **Resolución de conflictos**: Manejo de inconsistencias entre fuentes
- **Validación de integridad**: Verificación de claves y relaciones

#### 3.5 Formatear Datos
- **Estructura final**: Organización de datos para modelado
- **Tipos de datos**: Conversiones necesarias
- **Documentación**: Diccionario de datos actualizado

### Entregables
- Conjunto de datos preparado
- Descripción del conjunto de datos
- Reporte de preparación de datos

## Fase 4: Modelado

### Objetivo
Seleccionar y aplicar varias técnicas de modelado y calibrar sus parámetros a valores óptimos.

### Actividades Principales

#### 4.1 Seleccionar Técnica de Modelado
- **Evaluación de técnicas**: Algoritmos apropiados para el problema
- **Supuestos del modelo**: Requisitos y limitaciones
- **Justificación de selección**: Por qué elegir cada técnica

**Ejemplos por tipo de problema:**
```
Clasificación: Random Forest, SVM, Logistic Regression, XGBoost
Regresión: Linear Regression, Ridge, Lasso, Random Forest Regressor
Clustering: K-means, DBSCAN, Hierarchical Clustering
Series Temporales: ARIMA, LSTM, Prophet
```

#### 4.2 Generar Diseño de Prueba
- **División de datos**: Train/validation/test splits
- **Validación cruzada**: Estrategia de validación
- **Métricas de evaluación**: KPIs técnicos específicos

#### 4.3 Construir Modelo
- **Implementación**: Codificación y entrenamiento del modelo
- **Configuración de parámetros**: Valores iniciales
- **Documentación del proceso**: Pasos y decisiones tomadas

#### 4.4 Evaluar Modelo
- **Métricas de rendimiento**: Accuracy, precision, recall, F1, RMSE, etc.
- **Validación**: Resultados en datos de prueba
- **Interpretabilidad**: Comprensión de cómo funciona el modelo

**Ejemplo de evaluación:**
```python
from sklearn.metrics import classification_report, confusion_matrix

# Evaluación del modelo
y_pred = model.predict(X_test)
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(cm)
```

### Entregables
- Configuración de parámetros
- Modelos desarrollados
- Descripción del modelo
- Reporte de evaluación del modelo

## Fase 5: Evaluación

### Objetivo
Evaluar completamente el modelo y revisar los pasos ejecutados para construirlo, para asegurar que cumple los objetivos del negocio.

### Actividades Principales

#### 5.1 Evaluar Resultados
- **Evaluación de objetivos**: ¿Se cumplen los objetivos de data science?
- **Criterios de éxito**: Verificación contra métricas definidas
- **Aprobación del modelo**: Decisión go/no-go

#### 5.2 Revisar Proceso
- **Revisión metodológica**: ¿Se siguió correctamente CRISP-DM?
- **Lecciones aprendidas**: Qué funcionó y qué no
- **Mejoras posibles**: Optimizaciones para futuras iteraciones

#### 5.3 Determinar Próximos Pasos
- **Recomendaciones**: Implementar, iterar, o abandonar
- **Plan de implementación**: Si se aprueba el modelo
- **Identificación de mejoras**: Áreas de optimización

### Entregables
- Evaluación de resultados
- Modelos aprobados
- Revisión del proceso

## Fase 6: Implementación

### Objetivo
Organizar y presentar el conocimiento ganado de una forma que el cliente pueda usarlo.

### Actividades Principales

#### 6.1 Planificar Implementación
- **Plan de despliegue**: Cómo poner el modelo en producción
- **Plan de monitoreo**: Seguimiento del rendimiento
- **Plan de mantenimiento**: Actualizaciones y reentrenamiento

#### 6.2 Planificar Monitoreo y Mantenimiento
- **Métricas de monitoreo**: KPIs para seguimiento continuo
- **Alertas**: Cuándo y cómo detectar problemas
- **Cronograma de revisión**: Frecuencia de evaluaciones

#### 6.3 Producir Reporte Final
- **Resumen ejecutivo**: Resultados principales para stakeholders
- **Documentación técnica**: Detalles para el equipo técnico
- **Lecciones aprendidas**: Conocimiento para futuros proyectos

#### 6.4 Revisar Proyecto
- **Evaluación post-proyecto**: ¿Se cumplieron los objetivos?
- **Impacto empresarial**: Beneficios reales obtenidos
- **Recomendaciones futuras**: Próximos pasos estratégicos

### Entregables
- Plan de implementación
- Plan de monitoreo y mantenimiento
- Reporte final
- Revisión del proyecto

## Iteraciones y Flujo Cíclico

CRISP-DM es inherentemente iterativo. Es común volver a fases anteriores cuando:

- Los datos revelan nuevos insights que cambian la comprensión del negocio
- La calidad de los datos requiere reconsiderar los objetivos
- Los modelos no cumplen los criterios de éxito
- Nuevos requerimientos emergen durante la implementación

### Flujo Típico de Iteraciones

```
Business Understanding → Data Understanding → Data Preparation
        ↑                                            ↓
Implementation ← Evaluation ← Modeling ←──────────────┘
        ↓
    Monitoring → New Business Understanding (ciclo)
```

## Aplicación Práctica en Proyectos Académicos

### Para el Trabajo Final del Curso

1. **Documentar cada fase**: Crear entregables específicos por fase
2. **Justificar decisiones**: Explicar por qué se tomó cada decisión
3. **Mostrar iteraciones**: Documentar cómo evolucionó el proyecto
4. **Validar con datos**: Cada conclusión debe estar respaldada por evidencia

### Estructura Recomendada del Repositorio

```
proyecto/
├── 01_business_understanding/
│   ├── objetivos_negocio.md
│   ├── plan_proyecto.md
│   └── evaluacion_situacion.md
├── 02_data_understanding/
│   ├── exploratory_data_analysis.ipynb
│   ├── data_quality_report.md
│   └── data_description.md
├── 03_data_preparation/
│   ├── data_cleaning.ipynb
│   ├── feature_engineering.ipynb
│   └── final_dataset_description.md
├── 04_modeling/
│   ├── model_selection.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
├── 05_evaluation/
│   ├── business_evaluation.md
│   ├── technical_evaluation.md
│   └── process_review.md
├── 06_deployment/
│   ├── deployment_plan.md
│   ├── monitoring_plan.md
│   └── final_report.md
└── README.md
```

## Beneficios de Usar CRISP-DM

### Para Estudiantes
- **Estructura clara**: Marco organizado para proyectos complejos
- **Enfoque empresarial**: Conexión entre técnica y negocio
- **Documentación sistemática**: Historial completo del proyecto
- **Preparación profesional**: Metodología usada en la industria

### Para Proyectos
- **Reducción de riesgos**: Identificación temprana de problemas
- **Mejora de comunicación**: Lenguaje común entre equipos
- **Calidad del producto**: Proceso estructurado garantiza mejor resultado
- **Escalabilidad**: Metodología aplicable a proyectos de cualquier tamaño

## Herramientas y Tecnologías por Fase

### Comprensión del Negocio
- Herramientas de gestión: Trello, Jira, Asana
- Documentación: Markdown, LaTeX, Confluence
- Análisis de requisitos: Miro, Lucidchart

### Comprensión y Preparación de Datos
- **Python**: pandas, numpy, matplotlib, seaborn
- **R**: dplyr, ggplot2, tidyr
- **Bases de datos**: SQL, MongoDB
- **Herramientas de perfilado**: pandas-profiling, sweetviz

### Modelado
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: TensorFlow, PyTorch, Keras
- **Estadística**: statsmodels, scipy
- **AutoML**: H2O, AutoML, PyCaret

### Evaluación e Implementación
- **MLOps**: MLflow, Kubeflow, Neptune
- **Visualización**: Plotly, Dash, Streamlit
- **Monitoreo**: Prometheus, Grafana
- **Deploy**: Docker, Kubernetes, cloud platforms

## Ejemplo Completo: Proyecto de Predicción de Churn

### Fase 1: Comprensión del Negocio
```markdown
**Objetivo empresarial**: Reducir la pérdida de clientes en 20%
**Contexto**: Empresa de telecomunicaciones con alta rotación
**Criterio de éxito**: Identificar 80% de clientes que abandonarán
**Restricciones**: Presupuesto limitado, privacidad de datos
```

### Fase 2: Comprensión de Datos
```python
# Análisis inicial
import pandas as pd
df = pd.read_csv('telecom_churn.csv')
print(f"Registros: {len(df)}")
print(f"Variables: {df.shape[1]}")
print(f"Tasa de churn: {df['churn'].mean():.2%}")
```

### Fase 3: Preparación de Datos
```python
# Feature engineering
df['tenure_months'] = df['tenure'] / 30
df['total_charges_per_month'] = df['total_charges'] / df['tenure_months']
df['avg_monthly_charges'] = df['monthly_charges'].rolling(3).mean()
```

### Fase 4: Modelado
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1-score promedio: {scores.mean():.3f}")
```

### Fase 5: Evaluación
```markdown
**Resultados técnicos**: F1-score = 0.85, Precisión = 0.82
**Impacto empresarial**: Potencial reducción de churn del 15%
**Recomendación**: Implementar modelo con monitoreo mensual
```

### Fase 6: Implementación
```python
# Pipeline de producción
import joblib
joblib.dump(model, 'churn_model.pkl')

# Script de scoring
def predict_churn(customer_data):
    model = joblib.load('churn_model.pkl')
    return model.predict_proba(customer_data)[:, 1]
```

## Conclusión

CRISP-DM proporciona un marco robusto y probado para proyectos de data science. Su adopción asegura:

- **Orientación al negocio**: Soluciones que generan valor real
- **Calidad metodológica**: Proceso sistemático y documentado
- **Comunicación efectiva**: Lenguaje común entre equipos
- **Gestión de riesgos**: Identificación temprana de problemas
- **Escalabilidad**: Aplicable desde proyectos académicos hasta implementaciones empresariales

Para estudiantes, dominar CRISP-DM es esencial para desarrollar proyectos exitosos y prepararse para la práctica profesional en data science.

## Referencias y Recursos Adicionales

- [CRISP-DM 1.0 Step-by-step Data Mining Guide](https://www.kde.cs.uni-kassel.de/wp-content/uploads/lehre/ws2012-13/kdd/files/CRISP-DM_1.0_Step-by-step_data_mining_guide.pdf)
- [The CRISP-DM Process Model](https://www.datascience-pm.com/crisp-dm-2/)
- [KDD vs CRISP-DM vs SEMMA](https://www.kdnuggets.com/2014/10/crisp-dm-vs-kdd-vs-semma.html)
- [Modern Data Science Process](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview)
