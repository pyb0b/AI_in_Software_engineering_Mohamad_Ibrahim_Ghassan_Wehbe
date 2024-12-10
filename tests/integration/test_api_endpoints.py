import requests

BASE_URL = "http://127.0.0.1:8000"

def test_root():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json()["message"] == "API is running and monitoring is enabled!"

def test_recommend_endpoint():
    payload = {"user_id": 1, "model_type": "knn"}
    response = requests.post(f"{BASE_URL}/recommend/", json=payload)
    assert response.status_code == 200
    assert "recommendations" in response.json()
