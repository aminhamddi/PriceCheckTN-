"""API Testing Utilities"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_health():
    """Test API health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_api_prediction():
    """Test API prediction endpoint"""
    test_data = {
        "text": "This product is amazing! Best purchase ever!!!",
        "method": "bert_only"
    }

    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data

def test_api_model_info():
    """Test API model info endpoint"""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "version" in data
    assert "stage" in data