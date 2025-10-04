#!/usr/bin/env python3
"""
Startup script for the Time Series Training API
"""

import os
import sys

import uvicorn

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    print("🚀 Starting Time Series Training API...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔧 ReDoc Documentation: http://localhost:8000/redoc")
    print("❤️  Health Check: http://localhost:8000/health")
    print("=" * 60)

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
