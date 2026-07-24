"""Streamlit entrypoint for the Solar Project RAG app."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rag import SolarRAG


APP_BUILD = "2026-07-25-dataset-rebuild"


st.set_page_config(
    page_title="Solar Project RAG",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {
            background: rgba(255, 184, 0, 0.08);
            border: 1px solid rgba(255, 184, 0, 0.22);
            border-radius: 0.8rem;
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def initialize_rag_dataset_rebuild() -> SolarRAG:
    """Load the dataset once; ML training is deferred until the first query."""
    system = SolarRAG()
    system.setup()
    return system


try:
    rag = initialize_rag_dataset_rebuild()
except Exception as exc:
    st.error("The app could not load its dataset.")
    st.code(str(exc))
    st.info(
        "Make sure data/solar_dataset.csv or data/solar_dataset.csv.gz "
        "is committed to the GitHub repository."
    )
    st.stop()


def aqi_color(value: float) -> str:
    if value < 50:
        return "#2e9d50"
    if value < 100:
        return "#e3a008"
    return "#d14343"


def set_example(query: str) -> None:
    st.session_state["input_method"] = "Natural language"
    st.session_state["natural_query"] = query


st.title("☀️ Solar Project RAG")
st.caption(
    "Explore historical solar-generation and air-quality conditions across five Saudi cities."
)
st.caption(f"Build: {APP_BUILD}")

with st.sidebar:
    st.header("Query")
    mode = st.radio(
        "Input method",
        ["Structured", "Natural language"],
        key="input_method",
    )

    if mode == "Structured":
        selected_city = st.selectbox("City", rag.data_loader.cities)
        selected_date = st.date_input(
            "Date",
            value=rag.data_loader.min_date,
            min_value=rag.data_loader.min_date,
            max_value=rag.data_loader.max_date,
        )
        user_query = (
            f"Solar and air quality conditions in {selected_city} "
            f"on {selected_date:%Y-%m-%d}"
        )
    else:
        user_query = st.text_area(
            "Question",
            key="natural_query",
            placeholder="Solar conditions in Riyadh on 2024-01-15",
            height=120,
        )

    execute = st.button("Run query", type="primary", use_container_width=True)
    st.caption(
        f"Dataset: {rag.data_loader.min_date} to {rag.data_loader.max_date}"
    )


if execute:
    if not user_query.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Running retrieval and prediction… The first query trains compact models once."):
            result = rag.process_query(user_query)

        if result["status"] != "success":
            st.error(result.get("error", "The query could not be processed."))
        else:
            predictions = result["predictions"]
            aqi = predictions["aqi_value"]
            solar = predictions["solar_output_kwh"]
            risk = predictions["aqi_risk_level"]

            metric_1, metric_2, metric_3 = st.columns(3)
            metric_1.metric("Estimated solar output", f"{solar:.1f} kWh")
            metric_2.metric("Predicted AQI", f"{aqi:.0f}")
            metric_3.metric("Air-quality risk", risk)

            overview, conditions, insights = st.tabs(
                ["Overview", "Retrieved conditions", "RAG insights"]
            )

            with overview:
                chart = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=aqi,
                        title={"text": "Air Quality Index"},
                        gauge={
                            "axis": {"range": [0, 300]},
                            "bar": {"color": aqi_color(aqi)},
                            "steps": [
                                {"range": [0, 50], "color": "#d9f2df"},
                                {"range": [50, 100], "color": "#fff1c2"},
                                {"range": [100, 300], "color": "#f7d5d5"},
                            ],
                        },
                    )
                )
                chart.update_layout(height=360, margin=dict(l=30, r=30, t=70, b=20))
                st.plotly_chart(chart, use_container_width=True)
                st.success(result["llm_response"])

            with conditions:
                feature_labels = {
                    "temperature_2m_mean": "Mean temperature",
                    "relative_humidity_2m_mean": "Relative humidity",
                    "wind_speed_10m_mean": "Wind speed",
                    "cloud_cover_mean": "Cloud cover",
                    "precipitation_sum": "Precipitation",
                    "shortwave_radiation_sum": "Shortwave radiation",
                    "sunshine_duration": "Sunshine duration",
                    "pm10": "PM10",
                    "pm2_5": "PM2.5",
                    "carbon_monoxide": "Carbon monoxide",
                    "nitrogen_dioxide": "Nitrogen dioxide",
                    "ozone": "Ozone",
                    "sulphur_dioxide": "Sulphur dioxide",
                }
                rows = [
                    {"Feature": label, "Value": result["data"].get(key)}
                    for key, label in feature_labels.items()
                    if key in result["data"]
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            with insights:
                for insight in result["interpretations"]:
                    st.info(insight)

else:
    st.subheader("Try an example")
    examples = [
        ("Riyadh", "Solar conditions in Riyadh on 2024-01-15"),
        ("Jeddah", "Air quality and solar output in Jeddah on 2024-06-15"),
        ("Dammam", "Weather and solar conditions in Dammam on 2024-09-01"),
    ]
    columns = st.columns(len(examples))
    for column, (label, query) in zip(columns, examples):
        column.button(
            label,
            on_click=set_example,
            args=(query,),
            use_container_width=True,
        )

    st.info(
        "This app uses the repository's 2024 historical dataset. "
        "Choose a date in that range; it is not a live forecast service."
    )
