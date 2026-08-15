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
# Request schema
# --------------------------------------------------

class MachineInput(BaseModel):
    type: str = Field(
        ...,
        description="Machine type: L, M, or H"
    )

    air_temperature: float = Field(
        ...,
        description="Air temperature in Kelvin"
    )

    process_temperature: float = Field(
        ...,
        description="Process temperature in Kelvin"
    )

    rotational_speed: float = Field(
        ...,
        description="Rotational speed in RPM"
    )

    torque: float = Field(
        ...,
        description="Torque in Nm"
    )

    tool_wear: float = Field(
        ...,
        description="Tool wear in minutes"
    )


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

@app.post("/predict")
def predict(machine: MachineInput):

    try:

        machine_data = {
            "Type": machine.type,
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