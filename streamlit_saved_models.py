"""Simple Streamlit chat for the saved Project_Solar notebook models."""

from __future__ import annotations

import streamlit as st

from rag_saved_models import SavedModelSolarRAG


st.set_page_config(
    page_title="Solar IQ | Saved models",
    page_icon=":material/solar_power:",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_rag() -> SavedModelSolarRAG:
    """Load the dataset and persisted models once; never retrain at startup."""
    system = SavedModelSolarRAG()
    system.setup()
    return system


def aqi_label(value: float) -> str:
    """Convert the numeric US AQI estimate to its official display category."""
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Moderate"
    if value <= 150:
        return "Unhealthy for sensitive groups"
    if value <= 200:
        return "Unhealthy"
    if value <= 300:
        return "Very unhealthy"
    return "Hazardous"


def previous_context() -> dict:
    """Reuse the last city and date for short conversational follow-ups."""
    for message in reversed(st.session_state.messages):
        result = message.get("result")
        if result and result.get("status") == "success":
            return {
                "city": result["city"],
                "date": result["date"],
                "intents": result["intents"],
            }
    return {}


def show_result(result: dict) -> None:
    """Show only the decision information needed from one RAG answer."""
    if result.get("status") != "success":
        st.warning(result.get("error", "The analysis could not be completed."))
        return

    prediction = result["predictions"]
    risk = aqi_label(prediction["aqi_value"])
    summary = result["llm_response"].replace(
        f"({prediction['aqi_risk_level']})",
        f"({risk})",
    )

    st.caption(
        f"{result['city']} · {result['date']} · "
        f"{result.get('source_label', 'Model inputs')}"
    )
    st.write(summary)

    with st.container(horizontal=True):
        st.metric(
            "Solar energy",
            f"{prediction['solar_output_kwh']:.1f} kWh",
            border=True,
        )
        st.metric(
            "Air quality",
            risk,
            f"AQI {prediction['aqi_value']:.0f}",
            border=True,
        )
        st.metric("Model", "Saved notebook models", border=True)

    with st.expander("Decision guidance", icon=":material/strategy:"):
        for item in result["interpretations"]:
            st.markdown(f"- {item}")


rag = load_rag()
st.session_state.setdefault("messages", [])

st.title("Solar IQ · saved models", anchor=False)
st.caption(
    "A simple conversational RAG using the persisted models from "
    "`Project_Solar.ipynb`."
)
st.badge(
    "Saved models loaded",
    icon=":material/check_circle:",
    color="green",
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            show_result(message["result"])

suggestions = {
    "Riyadh solar": "Solar potential in Riyadh on February 2, 2024",
    "Jeddah suitability": "Was Jeddah suitable for solar on June 15, 2024?",
    "Dammam AQI": "Review air quality in Dammam on March 10, 2024",
}
selected = None
if not st.session_state.messages:
    selected = st.pills(
        "Suggested questions",
        list(suggestions),
        label_visibility="collapsed",
    )

prompt = st.chat_input(
    "Ask about a supported city and date…",
    submit_mode="disable",
)
if selected:
    prompt = suggestions[selected]

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running saved models…"):
            result = rag.process_query(prompt, context=previous_context())
        show_result(result)

    st.session_state.messages.append({"role": "assistant", "result": result})
