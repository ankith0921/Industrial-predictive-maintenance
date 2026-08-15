from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.service import predict_machine


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description=(
        "AI-powered predictive maintenance API using "
        "XGBoost and Isolation Forest."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Machine type
# --------------------------------------------------

class MachineType(str, Enum):
    L = "L"
    M = "M"
    H = "H"


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class MachineInput(BaseModel):

    type: MachineType = Field(
        ...,
        description="Machine type: L, M, or H"
    )

    air_temperature: float = Field(
        ...,
        gt=0,
        description="Air temperature in Kelvin"
    )

    process_temperature: float = Field(
        ...,
        gt=0,
        description="Process temperature in Kelvin"
    )

    rotational_speed: float = Field(
        ...,
        gt=0,
        description="Rotational speed in RPM"
    )

    torque: float = Field(
        ...,
        ge=0,
        description="Torque in Nm"
    )

    tool_wear: float = Field(
        ...,
        ge=0,
        description="Tool wear in minutes"
    )


# --------------------------------------------------
# Response schema
# --------------------------------------------------

class PredictionResponse(BaseModel):

    failure_probability: float
    predicted_failure: int

    anomaly_risk: float
    hybrid_risk: float

    risk_state: str
    recommended_action: str


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "industrial-predictive-maintenance"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(machine: MachineInput):

    try:

        machine_data = {
            "Type": machine.type.value,
            "Air temperature": machine.air_temperature,
            "Process temperature": machine.process_temperature,
            "Rotational speed": machine.rotational_speed,
            "Torque": machine.torque,
            "Tool wear": machine.tool_wear
        }

        result = predict_machine(
            machine_data
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )