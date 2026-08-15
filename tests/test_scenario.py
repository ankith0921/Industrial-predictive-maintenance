import pandas as pd

from src.scenarios.simulator import run_scenario


machine = {
    "Type": "L",
    "Air temperature": 298.1,
    "Process temperature": 308.6,
    "Rotational speed": 1551,
    "Torque": 42.8,
    "Tool wear": 0
}


def test_tool_wear_scenario():

    values = [
        0,
        50,
        100,
        150,
        200,
        250
    ]

    result = run_scenario(
        machine_data=machine,
        parameter="Tool wear",
        values=values
    )

    print("\nScenario Analysis:")
    print(result)

    assert isinstance(
        result,
        pd.DataFrame
    )

    assert len(result) == len(values)

    assert "Failure Probability" in result.columns
    assert "Anomaly Risk" in result.columns
    assert "Hybrid Risk" in result.columns
    assert "Risk State" in result.columns