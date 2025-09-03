# Metodología para Data Science 2025-2

Este repositorio contiene las guías y documentación esencial para el curso de Metodología para Data Science, incluyendo las mejores prácticas de desarrollo, validaciones de código y herramientas necesarias para el desarrollo de proyectos de data science.

## Contenido del Repositorio

Este repositorio está enfocado únicamente en la documentación y guías metodológicas, incluyendo:

- **Guía del Trabajo Final**: Estructura y requisitos para el proyecto final
- **Instalación de Python**: Instrucciones detalladas para configurar Python 3.12
- **Guía de Git**: Comandos básicos y flujo de trabajo con Git
- **Validaciones**: Estándares de calidad de código y pre-commit hooks
- **GitHub Actions**: Workflows de CI/CD para validación automática

## Estructura del Repositorio

```
.
├── .github/            # Configuraciones de GitHub (workflows, settings)
│   ├── workflows/     # GitHub Actions para CI/CD
│   └── settings.yml   # Configuraciones del repositorio
├── docs/              # Documentación del proyecto
├── .gitignore        # Archivos y directorios ignorados por git
├── README.md         # Este archivo
├── GUIA_TRABAJO_FINAL.md    # Guía para el desarrollo del trabajo final
├── PYTHON_INSTALLATION.md  # Instrucciones de instalación de Python
├── instrucciones_git.md     # Guía básica de Git
└── VALIDATIONS.md          # Estándares de calidad y validaciones
```

## Guías Incluidas

### 📋 [Guía del Trabajo Final](GUIA_TRABAJO_FINAL.md)
Estructura detallada para el desarrollo del proyecto final, incluyendo:
- Descripción del problema
- Metodología científica (CRISP-DM)
- Estructura de código
- Experimentación y validación
- Entregables por fase

### 🐍 [Instalación de Python](PYTHON_INSTALLATION.md)
Instrucciones paso a paso para instalar Python 3.12 en:
- Windows
- macOS
- Linux (Ubuntu/Debian, Fedora, CentOS/RHEL)
- Configuración de entornos virtuales

### 📚 [Guía de Git](instrucciones_git.md)
Comandos básicos y flujo de trabajo con Git:
- Instalación en diferentes sistemas operativos
- Configuración inicial
- Comandos básicos
- Trabajo con ramas
- Colaboración en equipo

### ✅ [Validaciones](VALIDATIONS.md)
Estándares de calidad de código:
- Pre-commit hooks
- Formateo de código (Black, isort)
- Linting (Flake8)
- Verificación de tipos (mypy)
- Documentación (docstrings)
- Tests y cobertura

## Configuración de Validaciones

Para configurar las validaciones de calidad de código en tu proyecto:

1. **Instalar pre-commit**:
   ```bash
   pip install pre-commit
   ```

2. **Configurar hooks** (crear `.pre-commit-config.yaml` en tu proyecto):
   ```bash
   pre-commit install
   ```

3. **Ejecutar validaciones**:
   ```bash
   # Ejecutar en todos los archivos
   pre-commit run --all-files
   
   # Se ejecutará automáticamente en cada commit
   ```

## GitHub Actions

El repositorio incluye workflows de GitHub Actions para:

- **Code Quality**: Validación automática de calidad de código
- **Branch Protection**: Protección de ramas principales
- **Documentation**: Generación automática de documentación

## Metodología CRISP-DM

Los proyectos deben seguir la metodología CRISP-DM:

1. **Comprensión del Negocio**
2. **Comprensión de los Datos**
3. **Preparación de los Datos**
4. **Modelado**
5. **Evaluación**
6. **Implementación**

Consulta la [Guía del Trabajo Final](GUIA_TRABAJO_FINAL.md) para más detalles.

## Convenciones de Código

### Estilo de Código
- Seguir PEP 8
- Usar Black para formateo automático
- Documentar con docstrings estilo Google
- Máxima longitud de línea: 88 caracteres

### Testing
- Tests unitarios obligatorios
- Cobertura mínima: 80%
- Estructura de tests organizada

### Documentación
- Docstrings obligatorios para funciones y clases
- README actualizado
- Comentarios claros en código complejo

## Flujo de Trabajo Git

1. **Crear rama**:
   ```bash
   git checkout -b feature/[tus-iniciales]/[descripcion]
   ```

2. **Realizar cambios**:
   ```bash
   git add .
   git commit -m "Descripción detallada de los cambios"
   ```

3. **Subir cambios**:
   ```bash
   git push origin feature/[tus-iniciales]/[descripcion]
   ```

4. **Crear Pull Request** y esperar revisión

## Contacto

Para dudas o consultas, contactar al instructor:
- GitHub: [@aladelca](https://github.com/aladelca)

---

## Licencia

Este material es para uso académico en el curso de Metodología para Data Science.
