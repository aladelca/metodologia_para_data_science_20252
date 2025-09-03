# Guía para el Trabajo Final: Metodología para Data Science

## Introducción

Este documento sirve como guía para el desarrollo de tu trabajo final en el curso de Metodología para Data Science. El objetivo es que apliques los conocimientos adquiridos en el curso para resolver un problema real utilizando metodología científica, desarrollo de código, experimentación y validación.

## Estructura del Trabajo Final

Tu trabajo final debe seguir una estructura similar a la de este repositorio, que incluye:

1. **Descripción del Problema**
2. **Metodología Científica**
3. **Código Fuente**
4. **Experimentación**
5. **Validación y Pruebas**

## 1. Descripción del Problema

### Objetivo

Define claramente el problema que vas a resolver. Incluye:

- **Contexto**: ¿Por qué es importante este problema?
- **Objetivos**: ¿Qué esperas lograr?
- **Hipótesis**: ¿Qué hipótesis estás probando?
- **Métricas de Éxito**: ¿Cómo medirás el éxito de tu solución?

### Ejemplo

En este repositorio, el problema es la optimización de estrategias de inversión con un presupuesto de 1,000,000 de soles peruanos y un horizonte de 10 años. Se evalúan diferentes opciones de inversión como mantener fondos en AFP, invertir en bonos del tesoro, comprar propiedades, invertir en S&P 500, diversificar en acciones e invertir en oro.

## 2. Metodología Científica

### Enfoque

Describe la metodología que utilizarás para abordar el problema. Incluye:

- **Recopilación de Datos**: ¿Qué datos necesitas y cómo los obtendrás?
- **Preprocesamiento**: ¿Cómo limpiarás y prepararás los datos?
- **Análisis Exploratorio**: ¿Qué técnicas usarás para entender los datos?
- **Modelado**: ¿Qué modelos o algoritmos implementarás?
- **Evaluación**: ¿Cómo evaluarás los resultados?

### Ejemplo

En este repositorio, la metodología incluye:
- Descarga de datos de rendimiento de AFP desde una API
- Limpieza y transformación de datos
- Análisis de diferentes estrategias de inversión
- Implementación de modelos predictivos
- Evaluación de resultados mediante pruebas y validación

## 3. Código Fuente

### Estructura del Código

Organiza tu código de manera clara y modular. Incluye:

- **Módulos**: Separa tu código en módulos lógicos (por ejemplo, preprocesamiento, modelado, evaluación)
- **Documentación**: Añade docstrings y comentarios explicativos
- **Control de Versiones**: Utiliza Git para gestionar tu código
- **Buenas Prácticas**: Sigue las mejores prácticas de programación (PEP 8, etc.)

### Ejemplo

En este repositorio, el código está organizado en:
- `src/pipeline/preprocesamiento.py`: Funciones para limpiar y transformar datos
- `src/pipeline/config.py`: Configuración y constantes
- `src/tests/`: Tests unitarios e integración

## 4. Experimentación

### Diseño de Experimentos

Describe cómo realizarás tus experimentos:

- **Configuración**: ¿Qué parámetros variarás?
- **Métricas**: ¿Qué métricas usarás para evaluar los resultados?
- **Análisis de Resultados**: ¿Cómo interpretarás los resultados?

### Ejemplo

En este repositorio, los experimentos incluyen:
- Comparación de diferentes estrategias de inversión
- Análisis de rendimiento a lo largo de 10 años
- Visualización de resultados mediante gráficos

## 5. Validación y Pruebas

### Pruebas

Implementa pruebas para validar tu código:

- **Tests Unitarios**: Prueba funciones individuales
- **Tests de Integración**: Prueba la interacción entre módulos
- **Validación de Datos**: Asegúrate de que los datos sean correctos

### Ejemplo

En este repositorio, se utilizan:
- `pytest` para ejecutar tests unitarios e integración
- `flake8` para verificar el estilo del código
- `mypy` para verificar tipos

## Herramientas y Tecnologías

### Git

- **Repositorio**: Crea un repositorio en GitHub o GitLab
- **Commits**: Realiza commits frecuentes y descriptivos
- **Branches**: Utiliza branches para desarrollar nuevas características
- **Pull Requests**: Solicita revisiones de código mediante pull requests

### Python

- **Entornos Virtuales**: Utiliza `venv` o `conda` para gestionar dependencias
- **Librerías**: Utiliza librerías como `pandas`, `numpy`, `scikit-learn`, etc.
- **Documentación**: Sigue la guía de estilo PEP 8

## Evaluación

La nota final se basará no solo en el repositorio final, sino también en:

- **Commits**: Frecuencia y calidad de los commits realizados
- **Pull Requests**: Número y calidad de los pull requests
- **Trabajo Colaborativo**: Participación activa en el desarrollo del proyecto
- **Documentación**: Claridad y completitud de la documentación

Es importante que trabajes colaborativamente, como si fuese un proyecto real de data science, demostrando habilidades de trabajo en equipo y comunicación efectiva.

## Recursos Adicionales

- [Documentación de Python](https://docs.python.org/3/)
- [Guía de Git](https://git-scm.com/doc)
- [PEP 8 - Guía de Estilo para Código Python](https://pep8.org/)
- [Tutorial de pytest](https://docs.pytest.org/en/stable/)

## Conclusión

Este trabajo final es una oportunidad para aplicar los conocimientos adquiridos en el curso y demostrar tu capacidad para resolver problemas reales utilizando metodología científica y buenas prácticas de desarrollo. Asegúrate de documentar todo el proceso y presentar tus resultados de manera clara y concisa.

---

Si tienes alguna pregunta o necesitas ayuda adicional, no dudes en contactar a tu instructor o consultar la documentación proporcionada.
