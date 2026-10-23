import pytest
from fastapi.testclient import TestClient


def test_health_endpoint():
    # model may not be trained; health should still respond
    from api.main import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_predict_requires_model():
    from api.main import app
    client = TestClient(app)
    payload = {
        "gender": "Male", "age": 67, "hypertension": 0, "heart_disease": 1,
        "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked",
    }
    r = client.post("/predict", json=payload)
    # either 200 (if model trained) or 503 (not trained) - both acceptable
    assert r.status_code in (200, 503)