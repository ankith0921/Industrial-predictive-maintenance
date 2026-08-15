from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# --------------------------------------------------
# Valid prediction
# --------------------------------------------------

def test_valid_prediction():

    payload = {
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "failure_probability" in data
    assert "predicted_failure" in data
    assert "anomaly_risk" in data
    assert "hybrid_risk" in data
    assert "risk_state" in data
    assert "recommended_action" in data


# --------------------------------------------------
# Invalid machine type
# --------------------------------------------------

def test_invalid_machine_type():

    payload = {
        "type": "X",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


# --------------------------------------------------
# Invalid temperature
# --------------------------------------------------

def test_invalid_temperature():

    payload = {
        "type": "L",
        "air_temperature": -10,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


# --------------------------------------------------
# Invalid rotational speed
# --------------------------------------------------

def test_invalid_rotational_speed():

    payload = {
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 0,
        "torque": 42.8,
        "tool_wear": 0
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422