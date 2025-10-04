#!/bin/bash
# Script de configuración obligatoria para desarrollo

set -e

echo "🔧 Configurando entorno de desarrollo..."

# Verificar que Python esté instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 no está instalado"
    exit 1
fi

# Instalar pre-commit si no está instalado
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Instalando pre-commit..."
    pip install pre-commit
fi

# Instalar hooks de pre-commit
echo "🪝 Instalando hooks de pre-commit..."
pre-commit install

# Verificar que el hook está instalado
if [ ! -f .git/hooks/pre-commit ]; then
    echo "❌ Error: El hook de pre-commit no se instaló correctamente"
    exit 1
fi

echo "✅ Pre-commit instalado y configurado correctamente"
echo ""
echo "⚠️  IMPORTANTE: Los hooks de pre-commit ahora se ejecutarán automáticamente"
echo "   antes de cada commit para garantizar la calidad del código."
echo ""
echo "   Para ejecutar manualmente: pre-commit run --all-files"
echo "   Para omitir (NO RECOMENDADO): git commit --no-verify"
