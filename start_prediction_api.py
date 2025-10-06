"""
Script para iniciar la API de Predicción
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("  Time Series Prediction API")
    print("=" * 60)
    print()
    print("Starting server...")
    print("API Documentation: http://localhost:8001/docs")
    print("API ReDoc: http://localhost:8001/redoc")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(
        "src.api.prediction_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
