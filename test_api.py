"""
Simple test script for the Training API
"""

import time

import requests  # type: ignore[import-untyped]


def test_api():
    """Test the Training API endpoints"""

    base_url = "http://localhost:8000"

    print("🧪 Testing Time Series Training API")
    print("=" * 50)

    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(
            "❌ Cannot connect to API. Make sure it's running on localhost:8000"
        )
        return

    # Test 2: Get available models
    print("\n2. Testing available models...")
    try:
        response = requests.get(f"{base_url}/models/available")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Available models: {models}")
        else:
            print(f"❌ Failed to get models: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 3: Get model info
    print("\n3. Testing model info...")
    try:
        response = requests.get(f"{base_url}/models/arima/info")
        if response.status_code == 200:
            info = response.json()
            print(f"✅ ARIMA info: {info['name']} - {info['description']}")
        else:
            print(f"❌ Failed to get model info: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 4: Training request (this will fail if no data file exists)
    print("\n4. Testing training request...")
    training_payload = {
        "model_types": ["arima"],
        "train_start_date": "2020-01-01",
        "train_end_date": "2023-12-31",
        "data_path": "s3://raw-data-stocks/stock_data/",
        "save_dir": "s3://raw-data-stocks/models/test-training",
        "arima_params": {
            "max_p": 3,
            "max_d": 1,
            "max_q": 3,
            "use_autorima": True,
        },
    }

    try:
        response = requests.post(f"{base_url}/train", json=training_payload)
        if response.status_code == 200:
            result = response.json()
            print("Training request successful")
            print(f"Job ID: {result['job_id']}")
            print(f"Status: {result['status']}")

            # Test 5: Check job status
            print("\n5. Testing job status...")
            job_id = result['job_id']

            # Wait a bit for processing
            time.sleep(1)

            status_response = requests.get(f"{base_url}/status/{job_id}")
            if status_response.status_code == 200:
                status = status_response.json()
                print("Job status retrieved")
                print(f"   Status: {status['status']}")
                print(f"   Progress: {status['progress_percentage']: .1f}%")
            else:
                print(f"❌ Failed to get status: {status_response.status_code}")

        else:
            print(f"❌ Train request failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 6: List jobs
    print("\n6. Testing job listing...")
    try:
        response = requests.get(f"{base_url}/jobs")
        if response.status_code == 200:
            jobs = response.json()
            print("Jobs listed successfully")
            print(f"   Total jobs: {len(jobs)}")
            for job in jobs:
                print(f"Job {job['job_id'][:8]}...: {job['status']}")
        else:
            print(f"❌ Failed to list jobs: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n🎉 API testing completed!")


if __name__ == "__main__":
    test_api()
