"""Business dashboard and conversational analyst for solar decision intelligence."""

from __future__ import annotations

import calendar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from rag import SolarRAG


APP_BUILD = "2026-07-25-project-dashboard"

SUGGESTIONS = {
    ":orange[:material/solar_power:] Solar potential": (
        "How much solar energy could Riyadh produce on February 2, 2026?"
    ),
    ":blue[:material/cloud:] Weather conditions": (
        "Review Jeddah's weather and solar potential on 15 June 2024."
    ),
    ":green[:material/health_and_safety:] Air-quality risk": (
        "How were the solar conditions and air-quality risk in Dammam "
        "on 10 March 2024?"
    ),
    ":violet[:material/location_city:] Site suitability": (
        "Was Medina suitable for solar generation on 1 September 2024?"
    ),
}

WEATHER_FEATURES = {
    "temperature_2m_mean": ("Mean temperature", "°C"),
    "relative_humidity_2m_mean": ("Relative humidity", "%"),
    "wind_speed_10m_mean": ("Wind speed", "km/h"),
    "cloud_cover_mean": ("Cloud cover", "%"),
    "precipitation_sum": ("Precipitation", "mm"),
    "shortwave_radiation_sum": ("Solar radiation", "MJ/m²"),
    "sunshine_duration": ("Sunshine duration", "hours"),
}

AIR_FEATURES = {
    "pm10": ("PM10", "µg/m³"),
    "pm2_5": ("PM2.5", "µg/m³"),
    "carbon_monoxide": ("Carbon monoxide", "µg/m³"),
    "nitrogen_dioxide": ("Nitrogen dioxide", "µg/m³"),
    "ozone": ("Ozone", "µg/m³"),
    "sulphur_dioxide": ("Sulphur dioxide", "µg/m³"),
}

st.set_page_config(
    page_title="SolarIQ | Solar intelligence for Saudi Arabia",
    page_icon=":material/solar_power:",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_FILE = Path(__file__).resolve().parent / "assets" / "solariq_theme.html"
st.html(THEME_FILE.read_text(encoding="utf-8"))


def get_secret(name: str, default: str = "") -> str:
    """Read a root-level Streamlit secret without displaying it."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value).strip()


@st.cache_resource(show_spinner=False)
def initialize_rag(
    ollama_api_key: str,
    ollama_host: str,
    ollama_model: str,
) -> SolarRAG:
    """Load the dataset and reusable prediction resources once."""
    system = SolarRAG(
        ollama_api_key=ollama_api_key,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
    )
    system.setup()
    return system


try:
    rag = initialize_rag(
        get_secret("OLLAMA_API_KEY"),
        get_secret("OLLAMA_HOST", "https://ollama.com"),
        get_secret("OLLAMA_MODEL", "gpt-oss:20b"),
    )
except Exception as exc:
    st.error(
        "Solar IQ could not load its analytical dataset.",
        icon=":material/error:",
    )
    st.code(str(exc))
    st.stop()


st.session_state.setdefault("messages", [])


def format_measurement(key: str, value: Any, unit: str) -> str:
    """Format measurements for executive-friendly tables."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if key == "sunshine_duration":
        number /= 3600
        return f"{number:.1f} {unit}"
    precision = 0 if unit == "%" else 1
    return f"{number:.{precision}f} {unit}".strip()


def measurement_table(data: Dict, features: Dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for key, (label, unit) in features.items():
        if data.get(key) is not None:
            rows.append(
                {
                    "Decision factor": label,
                    "Value": format_measurement(key, data[key], unit),
                }
            )
    return pd.DataFrame(rows)


def friendly_date(date_value: Optional[str]) -> str:
    if not date_value:
        return ""
    parsed = datetime.strptime(date_value, "%Y-%m-%d")
    return parsed.strftime("%B %d, %Y").replace(" 0", " ")


def conversation_context() -> Dict:
    """Use the last successful turn to resolve short follow-up questions."""
    for message in reversed(st.session_state.messages):
        result = message.get("result")
        if result and result.get("status") == "success":
            return {
                "city": result.get("city"),
                "date": result.get("date"),
                "intents": result.get("intents"),
            }
    return {}


def city_seasonal_profile(city: str) -> pd.DataFrame:
    """Build a monthly solar profile for the selected project market."""
    frame = rag.data_loader.df_merged
    if frame is None:
        return pd.DataFrame()
    city_frame = frame.loc[rag.data_loader._city_mask(city)].copy()
    if city_frame.empty:
        return pd.DataFrame()
    profile = (
        city_frame.groupby("Month", as_index=False)
        .agg(
            {
                "Estimated Daily Output (kWh)": "mean",
                "shortwave_radiation_sum": "mean",
            }
        )
        .sort_values("Month")
    )
    profile["Month"] = profile["Month"].map(
        lambda value: calendar.month_abbr[int(value)]
    )
    return profile.rename(
        columns={
            "Estimated Daily Output (kWh)": "Estimated solar output",
            "shortwave_radiation_sum": "Solar radiation",
        }
    )


def solar_position(value: float) -> tuple[str, str]:
    """Translate model output into a portfolio-relative business signal."""
    frame = rag.data_loader.df_merged
    target = "Estimated Daily Output (kWh)"
    if frame is None or target not in frame:
        return "Model estimate", "Portfolio benchmark unavailable"
    values = pd.to_numeric(frame[target], errors="coerce").dropna()
    if values.empty:
        return "Model estimate", "Portfolio benchmark unavailable"
    percentile = int(round((values <= value).mean() * 100))
    if percentile >= 75:
        label = "High potential"
    elif percentile >= 40:
        label = "Balanced"
    else:
        label = "Lower potential"
    return label, f"{percentile}th percentile in the modeled portfolio"


def render_analysis(result: Dict) -> None:
    """Render one business-facing analytical answer."""
    if result.get("status") != "success":
        st.write(result.get("error", "I could not complete that analysis."))
        return

    predictions = result["predictions"]
    data = result["data"]
    position_label, position_detail = solar_position(
        predictions["solar_output_kwh"]
    )

    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(
                "Executive outlook",
                anchor=False,
                width="stretch",
            )
            st.badge(
                result.get("source_kind", "model").capitalize(),
                icon=":material/database:",
                color="blue" if result.get("source_kind") == "historical" else "orange",
            )
        st.caption(
            f"{result['city']} · {friendly_date(result['date'])} · "
            f"{result.get('source_label', 'Model inputs')}"
        )
        st.markdown(result["llm_response"])

    st.subheader("Key results", anchor=False)
    with st.container(horizontal=True):
        st.metric(
            "Estimated solar energy",
            f"{predictions['solar_output_kwh']:.1f} kWh",
            border=True,
        )
        st.metric(
            "Portfolio position",
            position_label,
            position_detail,
            border=True,
        )
        st.metric(
            "Air quality",
            predictions["aqi_risk_level"],
            f"AQI {predictions['aqi_value']:.0f}",
            border=True,
        )

    with st.expander(
        "View decision drivers",
        icon=":material/analytics:",
    ):
        weather_column, air_column = st.columns(2)
        with weather_column:
            st.markdown("**Weather and solar**")
            st.dataframe(
                measurement_table(data, WEATHER_FEATURES),
                hide_index=True,
                width="stretch",
            )
        with air_column:
            st.markdown("**Air quality**")
            air_table = measurement_table(data, AIR_FEATURES)
            if air_table.empty:
                st.caption(
                    "Air-quality observations were unavailable. The estimate "
                    "uses the model's trained baseline."
                )
            else:
                st.dataframe(air_table, hide_index=True, width="stretch")

    with st.expander(
        "Analytical guidance",
        icon=":material/strategy:",
    ):
        for item in result["interpretations"]:
            st.markdown(f"- {item}")


with st.sidebar:
    st.html(
        """
        <div class="solariq-wordmark">
          <span class="solariq-brand-mark" aria-hidden="true"></span>
          <span class="solariq-brand-copy">
            <strong>Solar<b>IQ</b></strong>
            <small>Solar Intelligence · KSA</small>
          </span>
        </div>
        """
    )
    st.badge("Models operational", icon=":material/check_circle:", color="green")

    st.space("medium")
    selected_market = st.selectbox(
        "Market overview",
        rag.data_loader.cities,
        index=rag.data_loader.cities.index("Riyadh"),
    )
    st.badge(
        "Historical + live inputs",
        icon=":material/cloud_sync:",
        color="blue",
    )

    st.space("medium")
    st.caption(
        "Solar output · weather · air quality · suitability · combined analysis"
    )

    st.space("medium")
    if st.button(
        "Start a new analysis",
        icon=":material/add_comment:",
        type="primary",
        width="stretch",
        disabled=not st.session_state.messages,
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Decision engine · {APP_BUILD}")


with st.container(horizontal=True, vertical_alignment="center"):
    with st.container(width="stretch"):
        st.caption(":orange-badge[AI-POWERED DAILY FORECASTING]")
        st.title("Solar intelligence dashboard", anchor=False)
        st.write(
            "Explore solar generation, weather, air quality, and site suitability "
            "across the project's five Saudi markets."
        )
    st.badge(
        "5 markets",
        icon=":material/map:",
        color="orange",
    )


frame = rag.data_loader.df_merged
historical_days = int(frame["Date"].nunique()) if frame is not None else 0
with st.container(horizontal=True):
    st.metric(
        "Markets covered",
        str(len(rag.data_loader.cities)),
        "Saudi Arabia",
        border=True,
    )
    st.metric(
        "Historical coverage",
        f"{historical_days} days",
        "Full seasonal cycle",
        border=True,
    )
    st.metric(
        "Analytical models",
        "2 models",
        "Solar output + AQI",
        border=True,
    )


profile_column, brief_column = st.columns([1.65, 1], border=True)
with profile_column:
    st.subheader(f"{selected_market} seasonal profile", anchor=False)
    st.caption("Average modeled daily solar output by month · 2024 baseline")
    seasonal_profile = city_seasonal_profile(selected_market)
    if not seasonal_profile.empty:
        st.bar_chart(
            seasonal_profile,
            x="Month",
            y="Estimated solar output",
            x_label="Month",
            y_label="Daily solar output (kWh)",
        )

with brief_column:
    st.subheader("Project capabilities", anchor=False)
    st.caption("One clear view across the full analytical workflow")
    st.markdown(
        """
        **:material/solar_power: Solar generation**  
        Predict daily energy potential from environmental conditions.

        **:material/cloud: Weather and air quality**  
        Understand temperature, radiation, clouds, pollution, and AQI.

        **:material/task_alt: Site suitability**  
        Combine supporting factors, limitations, and practical guidance.
        """
    )
    st.caption("Use the analyst below for a city and date-specific answer.")


st.subheader("Ask the solar analyst", anchor=False)
st.caption(
    "Ask naturally about a market and date. Follow-ups such as "
    "“What about Jeddah?” retain the previous context."
)


for message in st.session_state.messages:
    with st.chat_message(
        message["role"],
        avatar=(
            ":material/business_messages:"
            if message["role"] == "assistant"
            else ":material/person:"
        ),
    ):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_analysis(message["result"])


selected_suggestion = None
if not st.session_state.messages:
    selected_suggestion = st.pills(
        "Suggested analyses",
        list(SUGGESTIONS),
        selection_mode="single",
        label_visibility="collapsed",
    )


prompt = st.chat_input(
    "Ask about solar potential, seasonal conditions, or environmental risk…",
    submit_mode="disable",
)
if selected_suggestion:
    prompt = SUGGESTIONS[selected_suggestion]


if prompt:
    context = conversation_context()
    context.setdefault("city", selected_market)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

    with st.chat_message(
        "assistant",
        avatar=":material/business_messages:",
    ):
        with st.spinner("Building the investment outlook…"):
            result = rag.process_query(prompt, context=context)
        render_analysis(result)

    st.session_state.messages.append({"role": "assistant", "result": result})
