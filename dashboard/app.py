import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("⚙️ Industrial Predictive Maintenance")

st.markdown(
    """
    **AI-powered machine health monitoring**

    Predict machine failure, detect abnormal operating
    conditions, and receive maintenance recommendations.
    """
)


st.divider()


# --------------------------------------------------
# Machine Input
# --------------------------------------------------

st.subheader("Machine Parameters")

col1, col2, col3 = st.columns(3)


with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temperature = st.number_input(
        "Air Temperature (K)",
        min_value=0.0,
        value=298.1,
        step=0.1
    )


with col2:

    process_temperature = st.number_input(
        "Process Temperature (K)",
        min_value=0.0,
        value=308.6,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed (RPM)",
        min_value=1.0,
        value=1551.0,
        step=1.0
    )


with col3:

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        value=42.8,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0.0,
        value=0.0,
        step=1.0
    )


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔍 Analyze Machine",
    type="primary",
    use_container_width=True
):

    payload = {
        "type": machine_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            st.session_state["prediction"] = result

        else:

            st.error(
                f"API error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the FastAPI server. "
            "Make sure the API is running."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The prediction request timed out."
        )


# --------------------------------------------------
# Results
# --------------------------------------------------

if "prediction" in st.session_state:

    result = st.session_state["prediction"]

    st.divider()

    st.subheader("Machine Health Analysis")

    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Failure Probability",
            f"{result['failure_probability'] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Anomaly Risk",
            f"{result['anomaly_risk'] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Hybrid Risk",
            f"{result['hybrid_risk'] * 100:.2f}%"
        )

    with col4:

        st.metric(
            "Prediction",
            "FAILURE"
            if result["predicted_failure"]
            else "NORMAL"
        )


    # ----------------------------------------------
    # Risk state
    # ----------------------------------------------

    risk_state = result["risk_state"]

    st.subheader("Risk Assessment")

    if risk_state == "HEALTHY":

        st.success(
            "🟢 HEALTHY — Machine operating normally."
        )

    elif risk_state == "ANOMALOUS":

        st.warning(
            "🟡 ANOMALOUS — Abnormal operating "
            "conditions detected."
        )

    elif risk_state == "PREDICTED FAILURE":

        st.warning(
            "🟠 PREDICTED FAILURE — Preventive "
            "maintenance recommended."
        )

    elif risk_state == "CRITICAL":

        st.error(
            "🔴 CRITICAL — Immediate inspection required."
        )


    # ----------------------------------------------
    # Maintenance recommendation
    # ----------------------------------------------

    st.subheader("Maintenance Recommendation")

    st.info(
        result["recommended_action"]
    )