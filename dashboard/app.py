import requests
import streamlit as st


# ============================================================
# Configuration
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #0b0f14;
    color: #e6edf3;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* Remove Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* Header */

.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #26303a;
    margin-bottom: 2rem;
}

.header-title {
    font-size: 1.8rem;
    font-weight: 600;
    color: #f0f3f6;
    letter-spacing: -0.02em;
}

.header-subtitle {
    font-size: 0.9rem;
    color: #8b949e;
    margin-top: 0.35rem;
}

.system-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #8b949e;
    font-size: 0.85rem;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #3fb950;
}


/* Section */

.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #f0f3f6;
    margin-bottom: 0.35rem;
}

.section-description {
    font-size: 0.85rem;
    color: #8b949e;
    margin-bottom: 1.2rem;
}


/* Metric cards */

.metric-card {
    background-color: #111820;
    border: 1px solid #26303a;
    border-radius: 10px;
    padding: 1.25rem;
    min-height: 125px;
}

.metric-label {
    color: #8b949e;
    font-size: 0.82rem;
    margin-bottom: 0.65rem;
}

.metric-value {
    color: #f0f3f6;
    font-size: 1.8rem;
    font-weight: 600;
    line-height: 1.2;
}

.metric-description {
    color: #6e7681;
    font-size: 0.74rem;
    margin-top: 0.55rem;
}


/* Risk */

.risk-box {
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-size: 0.95rem;
    font-weight: 500;
}

.risk-healthy {
    background-color: #0f2a1c;
    border: 1px solid #238636;
    color: #56d364;
}

.risk-anomalous {
    background-color: #2a210f;
    border: 1px solid #9e6a03;
    color: #e3b341;
}

.risk-predicted {
    background-color: #2b1b0f;
    border: 1px solid #d29922;
    color: #f2cc60;
}

.risk-critical {
    background-color: #2d1215;
    border: 1px solid #da3633;
    color: #ff7b72;
}


/* Recommendation */

.recommendation-box {
    background-color: #111820;
    border: 1px solid #26303a;
    border-radius: 10px;
    padding: 1.3rem 1.4rem;
    color: #c9d1d9;
    font-size: 0.95rem;
}


/* SHAP Explainability */

.shap-card {
    background-color: #111820;
    border: 1px solid #26303a;
    border-radius: 10px;
    padding: 1.4rem;
    margin-top: 0.5rem;
}

.shap-description {
    color: #8b949e;
    font-size: 0.85rem;
    margin-bottom: 1.4rem;
}

.shap-row {
    margin-bottom: 1rem;
}

.shap-feature {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
    color: #c9d1d9;
    font-size: 0.85rem;
}

.shap-impact {
    font-family: monospace;
    font-size: 0.8rem;
    color: #8b949e;
}

.shap-bar-container {
    width: 100%;
    height: 8px;
    background-color: #1b222c;
    border-radius: 4px;
    overflow: hidden;
}

.shap-bar-negative {
    height: 100%;
    background-color: #3b82f6;
    border-radius: 4px;
}

.shap-bar-positive {
    height: 100%;
    background-color: #f85149;
    border-radius: 4px;
}

.shap-legend {
    display: flex;
    gap: 1.5rem;
    margin-top: 1.2rem;
    color: #8b949e;
    font-size: 0.75rem;
}

.shap-legend-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.shap-legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.shap-negative-dot {
    background-color: #3b82f6;
}

.shap-positive-dot {
    background-color: #f85149;
}


/* Streamlit inputs */

label {
    color: #c9d1d9 !important;
}


/* Analyze button */

.stButton > button {
    width: 100%;
    min-height: 42px;
    background-color: #238636;
    color: #ffffff;
    border: 1px solid #2ea043;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #2ea043;
    border-color: #3fb950;
}


/* Divider */

hr {
    border-color: #26303a;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.html(
    """
    <div class="header-container">

        <div>

            <div class="header-title">
                Industrial Predictive Maintenance
            </div>

            <div class="header-subtitle">
                AI-powered machine health monitoring and risk assessment
            </div>

        </div>

        <div class="system-status">

            <span class="status-dot"></span>

            Prediction API Online

        </div>

    </div>
    """
)


# ============================================================
# Machine Parameters
# ============================================================

st.html(
    """
    <div class="section-title">
        Machine Parameters
    </div>

    <div class="section-description">
        Enter the current operating conditions of the machine.
    </div>
    """
)


col1, col2, col3 = st.columns(3)


with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"],
    )

    air_temperature = st.number_input(
        "Air Temperature (K)",
        min_value=0.0,
        value=298.1,
        step=0.1,
    )


with col2:

    process_temperature = st.number_input(
        "Process Temperature (K)",
        min_value=0.0,
        value=308.6,
        step=0.1,
    )

    rotational_speed = st.number_input(
        "Rotational Speed (RPM)",
        min_value=1.0,
        value=1551.0,
        step=1.0,
    )


with col3:

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        value=42.8,
        step=0.1,
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )


st.write("")


# ============================================================
# Analyze Machine
# ============================================================

if st.button("Analyze Machine"):

    payload = {
        "type": machine_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:

            st.session_state["prediction"] = response.json()

        else:

            st.error(
                f"Prediction API returned status code "
                f"{response.status_code}."
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the prediction API."
        )

    except requests.exceptions.Timeout:

        st.error(
            "Prediction request timed out."
        )


# ============================================================
# Results
# ============================================================

if "prediction" in st.session_state:

    result = st.session_state["prediction"]

    st.divider()


    # ========================================================
    # Machine Health
    # ========================================================

    st.html(
        """
        <div class="section-title">
            Machine Health
        </div>
        """
    )


    metric1, metric2, metric3, metric4 = st.columns(4)


    # --------------------------------------------------------
    # Failure Probability
    # --------------------------------------------------------

    with metric1:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Failure Probability
                </div>

                <div class="metric-value">
                    {result["failure_probability"] * 100:.2f}%
                </div>

                <div class="metric-description">
                    XGBoost failure prediction
                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # Anomaly Risk
    # --------------------------------------------------------

    with metric2:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Anomaly Risk
                </div>

                <div class="metric-value">
                    {result["anomaly_risk"] * 100:.2f}%
                </div>

                <div class="metric-description">
                    Isolation Forest anomaly score
                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # Hybrid Risk
    # --------------------------------------------------------

    with metric3:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Hybrid Risk
                </div>

                <div class="metric-value">
                    {result["hybrid_risk"] * 100:.2f}%
                </div>

                <div class="metric-description">
                    Combined model risk
                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with metric4:

        prediction_label = (
            "FAILURE"
            if result["predicted_failure"]
            else "NORMAL"
        )

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Prediction
                </div>

                <div class="metric-value">
                    {prediction_label}
                </div>

                <div class="metric-description">
                    XGBoost classification
                </div>

            </div>
            """
        )


    st.write("")


    # ========================================================
    # Risk Assessment
    # ========================================================

    st.html(
        """
        <div class="section-title">
            Risk Assessment
        </div>
        """
    )


    risk_state = result["risk_state"]


    if risk_state == "HEALTHY":

        st.html(
            """
            <div class="risk-box risk-healthy">
                HEALTHY — Machine operating normally.
            </div>
            """
        )


    elif risk_state == "ANOMALOUS":

        st.html(
            """
            <div class="risk-box risk-anomalous">
                ANOMALOUS — Abnormal operating conditions detected.
            </div>
            """
        )


    elif risk_state == "PREDICTED FAILURE":

        st.html(
            """
            <div class="risk-box risk-predicted">
                PREDICTED FAILURE — Preventive maintenance recommended.
            </div>
            """
        )


    elif risk_state == "CRITICAL":

        st.html(
            """
            <div class="risk-box risk-critical">
                CRITICAL — Immediate inspection required.
            </div>
            """
        )


    st.write("")


    # ========================================================
    # Operating Conditions
    # ========================================================

    st.subheader("Operating Conditions")


    condition1, condition2, condition3 = st.columns(3)
    condition4, condition5, condition6 = st.columns(3)


    with condition1:

        st.metric(
            "Machine Type",
            machine_type
        )


    with condition2:

        st.metric(
            "Air Temperature",
            f"{air_temperature:.1f} K"
        )


    with condition3:

        st.metric(
            "Process Temperature",
            f"{process_temperature:.1f} K"
        )


    with condition4:

        st.metric(
            "Rotational Speed",
            f"{rotational_speed:.0f} RPM"
        )


    with condition5:

        st.metric(
            "Torque",
            f"{torque:.1f} Nm"
        )


    with condition6:

        st.metric(
            "Tool Wear",
            f"{tool_wear:.0f} min"
        )


    st.write("")


    # ========================================================
    # Maintenance Recommendation
    # ========================================================

    st.html(
        """
        <div class="section-title">
            Maintenance Recommendation
        </div>
        """
    )


    st.html(
        f"""
        <div class="recommendation-box">
            {result["recommended_action"]}
        </div>
        """
    )


    st.write("")


    # ========================================================
    # Model Explainability - SHAP
    # ========================================================

    st.html(
        """
        <div class="section-title">
            Model Explainability
        </div>

        <div class="section-description">
            Factors that influenced the XGBoost prediction.
        </div>
        """
    )


    shap_explanation = result.get(
        "shap_explanation",
        []
    )


    if shap_explanation:

        # ----------------------------------------------------
        # Sort features by absolute SHAP impact
        # ----------------------------------------------------

        shap_explanation = sorted(
            shap_explanation,
            key=lambda x: abs(float(x["impact"])),
            reverse=True
        )


        max_impact = max(
            abs(float(item["impact"]))
            for item in shap_explanation
        )


        shap_html = """
        <div class="shap-card">
        """


        shap_html += """
        <div class="shap-description">
            Negative values push the prediction toward lower
            failure risk. Positive values push the prediction
            toward higher failure risk.
        </div>
        """


        # ----------------------------------------------------
        # SHAP feature rows
        # ----------------------------------------------------

        for item in shap_explanation:

            feature = item["feature"]

            impact = float(
                item["impact"]
            )


            if max_impact > 0:

                bar_width = (
                    abs(impact) / max_impact
                ) * 100

            else:

                bar_width = 0


            # ------------------------------------------------
            # Negative impact
            # ------------------------------------------------

            if impact < 0:

                bar_html = f"""
                <div class="shap-bar-container">

                    <div
                        class="shap-bar-negative"
                        style="width: {bar_width:.1f}%"
                    ></div>

                </div>
                """


            # ------------------------------------------------
            # Positive impact
            # ------------------------------------------------

            else:

                bar_html = f"""
                <div class="shap-bar-container">

                    <div
                        class="shap-bar-positive"
                        style="width: {bar_width:.1f}%"
                    ></div>

                </div>
                """


            shap_html += f"""
            <div class="shap-row">

                <div class="shap-feature">

                    <span>
                        {feature}
                    </span>

                    <span class="shap-impact">
                        {impact:+.3f}
                    </span>

                </div>

                {bar_html}

            </div>
            """


        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        shap_html += """
            <div class="shap-legend">

                <div class="shap-legend-item">

                    <span
                        class="shap-legend-dot
                               shap-negative-dot">
                    </span>

                    Lower failure risk

                </div>


                <div class="shap-legend-item">

                    <span
                        class="shap-legend-dot
                               shap-positive-dot">
                    </span>

                    Higher failure risk

                </div>

            </div>

        </div>
        """


        st.html(
            shap_html
        )


    else:

        st.info(
            "SHAP explanation is not available for this prediction."
        )