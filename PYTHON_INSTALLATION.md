# Guía de Instalación de Python 3.12

Esta guía proporciona instrucciones detalladas para instalar Python 3.12 en diferentes sistemas operativos.

## Contenido
- [Windows](#windows)
- [macOS](#macos)
- [Linux](#linux)
- [Verificación de la instalación](#verificación-de-la-instalación)
- [Entornos virtuales](#entornos-virtuales)

## Windows

1. **Descargar el instalador**:
   - Visita la página oficial de Python: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Haz clic en "Download Python 3.12.x"
   - Selecciona el instalador de 64 bits para Windows

2. **Ejecutar el instalador**:
   - Marca la casilla "Add Python 3.12 to PATH"
   - Selecciona "Install Now" para una instalación estándar
   - Opcionalmente, puedes elegir "Customize installation" para opciones avanzadas

3. **Verificar la instalación**:
   - Abre el Command Prompt (cmd) o PowerShell
   - Ejecuta los siguientes comandos:
   ```
   python --version
   pip --version
   ```

## macOS

### Usando Homebrew (Recomendado)

1. **Instalar Homebrew** (si no lo tienes):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Instalar Python 3.12**:
   ```bash
   brew update
   brew install python@3.12
   ```

3. **Agregar Python a tu PATH**:
   ```bash
   echo 'export PATH="/usr/local/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
   # O si usas bash:
   # echo 'export PATH="/usr/local/opt/python@3.12/bin:$PATH"' >> ~/.bash_profile
   ```

4. **Actualizar tu terminal**:
   ```bash
   source ~/.zshrc  # o ~/.bash_profile si usas bash
   ```

### Usando el instalador oficial

1. **Descargar el instalador**:
   - Visita [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Haz clic en "Download Python 3.12.x"
   - Selecciona el instalador macOS

2. **Ejecutar el instalador**:
   - Sigue las instrucciones en pantalla
   - Completa la instalación

3. **Verificar la instalación**:
   ```bash
   python3 --version
   pip3 --version
   ```

## Linux

### Ubuntu/Debian

1. **Actualizar los repositorios**:
   ```bash
   sudo apt update
   sudo apt upgrade
   ```

2. **Instalar dependencias**:
   ```bash
   sudo apt install software-properties-common
   ```

3. **Agregar el PPA de deadsnakes** (repositorio con versiones recientes de Python):
   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   ```

4. **Instalar Python 3.12**:
   ```bash
   sudo apt install python3.12 python3.12-venv python3.12-dev
   ```

5. **Instalar pip**:
   ```bash
   curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.12
   ```

### Fedora

1. **Instalar Python 3.12**:
   ```bash
   sudo dnf install python3.12
   ```

2. **Instalar pip y herramientas de desarrollo**:
   ```bash
   sudo dnf install python3.12-pip python3.12-devel
   ```

### Arch Linux

1. **Instalar Python 3.12**:
   ```bash
   sudo pacman -S python
   ```
   En Arch, el paquete `python` siempre se refiere a la última versión estable.

## Verificación de la instalación

Verifica que Python 3.12 esté correctamente instalado ejecutando:

```bash
# En Windows:
python --version
pip --version

# En macOS/Linux:
python3.12 --version
pip3.12 --version
```

También puedes verificar que puedes ejecutar el intérprete interactivo:

```bash
# En Windows:
python

# En macOS/Linux:
python3.12
```

Deberías ver algo como:
```
Python 3.12.x (tags/...) [MSC v... (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

Para salir del intérprete, escribe `exit()` o presiona Ctrl+Z en Windows o Ctrl+D en macOS/Linux.

## Entornos virtuales

Es recomendable usar entornos virtuales para cada proyecto. Aquí te mostramos cómo:

1. **Crear un entorno virtual**:
   ```bash
   # En Windows:
   python -m venv myenv

   # En macOS/Linux:
   python3.12 -m venv myenv
   ```

2. **Activar el entorno virtual**:
   ```bash
   # En Windows (Command Prompt):
   myenv\Scripts\activate

   # En Windows (PowerShell):
   myenv\Scripts\Activate.ps1

   # En macOS/Linux:
   source myenv/bin/activate
   ```

3. **Desactivar el entorno virtual**:
   ```bash
   deactivate
   ```

## Recursos adicionales

- [Documentación oficial de Python](https://docs.python.org/3.12/)
- [Tutorial de Python](https://docs.python.org/3.12/tutorial/index.html)
- [PEP 8 - Guía de estilo para código Python](https://pep8.org/)

---

Si encuentras algún problema durante la instalación, consulta la [documentación oficial](https://docs.python.org/3.12/using/index.html) o busca ayuda en los [foros de Python](https://discuss.python.org/).
