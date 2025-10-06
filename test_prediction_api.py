"""
Script de prueba rápida para verificar la API de predicción
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8001"


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_health():
    """Test health endpoint"""
    print_section("1. Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_get_models():
    """Test get models endpoint"""
    print_section("2. Get Available Models")
    try:
        response = requests.get(f"{BASE_URL}/models")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"\nTotal Models: {data['total_models']}")
        print(f"Models Directory: {data['models_directory']}")
        print(f"\nAvailable Models:")
        for model in data['available_models']:
            print(f"  - {model}")
        
        if data['model_details']:
            print(f"\nModel Details:")
            for detail in data['model_details']:
                print(f"  {detail['model_name']}:")
                print(f"    Type: {detail['model_type']}")
                print(f"    Size: {detail['file_size_mb']} MB")
        
        return response.status_code == 200, data.get('available_models', [])
    except Exception as e:
        print(f"Error: {e}")
        return False, []


def test_predict_single_model(model_name):
    """Test prediction with single model"""
    print_section(f"3. Predict with {model_name}")
    try:
        payload = {
            "model_name": model_name,
            "forecast_days": 7,
            "include_confidence_intervals": True
        }
        
        print(f"\nRequest Payload:")
        print(json.dumps(payload, indent=2))
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nRequest ID: {data['request_id']}")
            print(f"Status: {data['status']}")
            print(f"Forecast Period: {data['forecast_start_date']} to {data['forecast_end_date']}")
            print(f"Total Models Used: {data['total_models_used']}")
            print(f"Prediction Time: {data['total_prediction_time_seconds']}s")
            
            if data['predictions']:
                pred = data['predictions'][0]
                print(f"\n{pred['model_name']} Results:")
                print(f"  Model Type: {pred['model_type']}")
                print(f"  Predictions (first 3): {pred['predictions'][:3]}")
                print(f"  Dates (first 3): {pred['dates'][:3]}")
                if pred['confidence_lower']:
                    print(f"  Has confidence intervals: Yes")
            
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_predict_all_models():
    """Test prediction with all models"""
    print_section("4. Predict with All Models")
    try:
        payload = {
            "forecast_days": 7,
            "ensemble_method": "mean",
            "include_confidence_intervals": False
        }
        
        print(f"\nRequest Payload:")
        print(json.dumps(payload, indent=2))
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nRequest ID: {data['request_id']}")
            print(f"Status: {data['status']}")
            print(f"Total Models Used: {data['total_models_used']}")
            print(f"Successful: {data['successful_predictions']}")
            print(f"Failed: {data['failed_predictions']}")
            print(f"Prediction Time: {data['total_prediction_time_seconds']}s")
            
            print(f"\nModels Used:")
            for pred in data['predictions']:
                print(f"  - {pred['model_name']} ({pred['model_type']})")
            
            if data['ensemble_prediction']:
                print(f"\nEnsemble Prediction (first 3): {data['ensemble_prediction'][:3]}")
            
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  Time Series Prediction API - Quick Test")
    print("=" * 60)
    print(f"\nTesting API at: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: Health Check
    results.append(("Health Check", test_health()))
    
    # Test 2: Get Models
    success, models = test_get_models()
    results.append(("Get Models", success))
    
    # Test 3: Predict with Single Model (if available)
    if models:
        first_model = models[0]
        results.append((f"Predict ({first_model})", test_predict_single_model(first_model)))
    
    # Test 4: Predict with All Models
    if models:
        results.append(("Predict All Models", test_predict_all_models()))
    
    # Summary
    print_section("Test Summary")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed successfully!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
    
    print("=" * 60)


if __name__ == "__main__":
    print("\nMake sure the API server is running on http://localhost:8001")
    print("You can start it with: python start_prediction_api.py")
    input("\nPress Enter to run tests...")
    
    main()
