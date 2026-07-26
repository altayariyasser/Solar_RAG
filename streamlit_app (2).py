"""SolarIQ dashboard with the project's Python RAG chat integration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import streamlit as st
import streamlit.components.v1 as components

from rag import SolarRAG


APP_DIR = Path(__file__).resolve().parent
HTML_FILE = APP_DIR / "solariq_standalone.html"

SUGGESTIONS = [
    "How much solar energy will Riyadh produce tomorrow?",
    "Review Jeddah's solar conditions on 15 June 2024.",
    "What was the air-quality risk in Dammam on 10 March 2024?",
]

st.set_page_config(
    page_title="SolarIQ — Solar Intelligence for Saudi Arabia",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
      --orange: #FDB813;
      --orange2: #FF8A00;
      --yellow: #FFD54A;
      --ink: #0A0E1A;
      --ink2: #101628;
      --text-1: #F4F6FB;
      --text-2: #B9C2D8;
      --text-3: #7D879C;
      --green: #38E27C;
      --red: #FF6B6B;
      --glass: rgba(13, 18, 34, 0.62);
      --glass-border: rgba(255, 255, 255, 0.09);
      --shadow: 0 24px 60px -18px rgba(0, 0, 0, 0.55);
    }

    html, body, [class*="st-"], [class*="css"] {
      font-family: 'Inter', system-ui, sans-serif;
    }

    #MainMenu,
    header,
    footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
      display: none !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .stApp {
      background:
        radial-gradient(circle at 92% 8%, rgba(255,138,0,.14), transparent 28%),
        linear-gradient(180deg, #0A0E1A 0%, #14101D 55%, #24151B 100%) !important;
      color: var(--text-1);
    }

    .block-container,
    [data-testid="stMainBlockContainer"] {
      max-width: 100% !important;
      padding: 0 0 72px !important;
    }

    iframe {
      width: 100%;
      border: 0;
      display: block;
    }

    .rag-heading {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 2px;
    }

    .rag-heading h2 {
      margin: 0;
      color: var(--text-1);
      font-family: 'Orbitron';
      font-size: 16px;
      font-weight: 900;
      letter-spacing: 1.5px;
      text-transform: uppercase;
    }

    .rag-engine {
      color: var(--orange);
      background: rgba(253,184,19,.10);
      border: 1px solid rgba(253,184,19,.35);
      border-radius: 100px;
      padding: 7px 15px;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 1.4px;
      text-transform: uppercase;
    }

    .rag-description {
      color: var(--text-2);
      font-size: 13px;
      line-height: 1.7;
      margin: 0 0 12px;
    }

    .st-key-rag_chat {
      width: min(1440px, calc(100% - 80px));
      margin: 0 auto;
    }

    .st-key-rag_chat > div > [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-rag_chat [data-testid="stVerticalBlockBorderWrapper"] {
      padding: 30px 32px 26px;
      color: var(--text-1);
      background: var(--glass);
      border: 1px solid rgba(253,184,19,.30) !important;
      border-radius: 24px !important;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }

    .st-key-rag_question [data-baseweb="input"] > div {
      min-height: 58px;
      color: var(--text-1);
      background: rgba(255,255,255,.055);
      border: 1px solid var(--glass-border);
      border-radius: 100px;
      box-shadow: none;
    }

    .st-key-rag_question input {
      color: var(--text-1) !important;
      padding: 0 22px !important;
    }

    .st-key-rag_question input::placeholder {
      color: var(--text-3) !important;
    }

    .st-key-rag_chat [data-testid="stFormSubmitButton"] button {
      min-height: 58px;
      color: #241500 !important;
      background: linear-gradient(135deg, var(--yellow), var(--orange2)) !important;
      border: 0 !important;
      border-radius: 100px !important;
      font-weight: 800 !important;
      box-shadow: 0 14px 34px -10px rgba(253,150,19,.65);
    }

    .st-key-rag_suggestion_0 button,
    .st-key-rag_suggestion_1 button,
    .st-key-rag_suggestion_2 button,
    .st-key-new_rag_chat button {
      min-height: 38px;
      color: var(--text-2) !important;
      background: rgba(255,255,255,.045) !important;
      border: 1px solid var(--glass-border) !important;
      border-radius: 100px !important;
      font-size: 12px !important;
      font-weight: 700 !important;
    }

    .st-key-rag_suggestion_0 button:hover,
    .st-key-rag_suggestion_1 button:hover,
    .st-key-rag_suggestion_2 button:hover,
    .st-key-new_rag_chat button:hover {
      color: var(--text-1) !important;
      border-color: rgba(253,184,19,.45) !important;
    }

    [data-testid="stChatMessage"] {
      margin-top: 12px;
      padding: 18px 20px;
      color: var(--text-1);
      background: rgba(255,255,255,.045);
      border: 1px solid var(--glass-border);
      border-radius: 16px;
    }

    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li {
      color: var(--text-1);
    }

    [data-testid="stMetric"] {
      min-height: 112px;
      padding: 18px;
      background: rgba(255,255,255,.045);
      border: 1px solid var(--glass-border);
      border-radius: 16px;
    }

    [data-testid="stMetricLabel"] p {
      color: var(--text-3) !important;
      font-size: 10px !important;
      font-weight: 800 !important;
      letter-spacing: 1.1px;
      text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
      color: var(--orange) !important;
      font-family: 'Orbitron' !important;
      font-size: 24px !important;
      font-weight: 900 !important;
    }

    [data-testid="stMetricDelta"] {
      color: var(--text-2) !important;
    }

    [data-testid="stExpander"] {
      color: var(--text-1);
      background: rgba(255,255,255,.035);
      border: 1px solid var(--glass-border) !important;
      border-radius: 14px !important;
      overflow: hidden;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details {
      color: var(--text-1) !important;
      background: transparent !important;
    }

    [data-testid="stAlert"] {
      color: var(--text-1);
      background: rgba(13,18,34,.88);
      border: 1px solid var(--glass-border);
      border-radius: 14px;
    }

    .rag-meta {
      color: var(--text-3);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .5px;
      margin-top: -6px;
    }

    @media (max-width: 760px) {
      .st-key-rag_chat {
        width: calc(100% - 24px);
      }
      .st-key-rag_chat > div > [data-testid="stVerticalBlockBorderWrapper"],
      .st-key-rag_chat [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 22px 18px;
      }
      .rag-heading {
        align-items: flex-start;
        flex-direction: column;
      }
      .rag-engine {
        font-size: 9px;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(name: str, default: str = "") -> str:
    """Read an optional root-level Streamlit secret."""
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
    """Load the dataset and reusable RAG resources once."""
    system = SolarRAG(
        ollama_api_key=ollama_api_key,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
    )
    system.setup()
    return system


def conversation_context(messages: list[Dict]) -> Dict:
    """Keep city, date, and intent context for short follow-up questions."""
    for message in reversed(messages):
        result = message.get("result")
        if result and result.get("status") == "success":
            return {
                "city": result.get("city"),
                "date": result.get("date"),
                "intents": result.get("intents"),
            }
    return {}


def render_rag_result(result: Dict) -> None:
    """Render one answer from the project's Python RAG pipeline."""
    if result.get("status") != "success":
        st.error(result.get("error", "The RAG analysis could not be completed."))
        return

    predictions = result["predictions"]
    st.markdown(result["llm_response"])
    st.markdown(
        (
            f"<p class='rag-meta'>{result['city']} · {result['date']} · "
            f"{result.get('source_label', 'Model inputs')}</p>"
        ),
        unsafe_allow_html=True,
    )

    solar_column, aqi_column, risk_column = st.columns(3)
    with solar_column:
        st.metric(
            "Estimated solar energy",
            f"{predictions['solar_output_kwh']:.1f} kWh",
        )
    with aqi_column:
        st.metric(
            "Estimated AQI",
            f"{predictions['aqi_value']:.0f}",
        )
    with risk_column:
        st.metric(
            "Air-quality risk",
            predictions["aqi_risk_level"],
        )

    with st.expander("Retrieved knowledge used in this answer"):
        for item in result["interpretations"]:
            st.markdown(f"- {item}")

    if result.get("llm_error"):
        st.caption(
            "The cloud explanation was unavailable, so SolarIQ returned its "
            "grounded local summary."
        )


if not HTML_FILE.exists():
    st.error("The SolarIQ interface file `solariq_standalone.html` is missing.")
    st.stop()

components.html(
    HTML_FILE.read_text(encoding="utf-8"),
    height=1420,
    scrolling=True,
)

st.session_state.setdefault("rag_messages", [])

with st.container(key="rag_chat", border=True):
    st.markdown(
        """
        <div class="rag-heading">
          <h2>Ask SolarIQ</h2>
          <span class="rag-engine">parse → retrieve → predict → ground</span>
        </div>
        <p class="rag-description">
          Ask in plain English. SolarIQ identifies the city, date, and topic,
          retrieves historical or live conditions, runs your trained models,
          and grounds the answer in your knowledge base.
        </p>
        """,
        unsafe_allow_html=True,
    )

    prompt = None
    with st.form("rag_question_form", clear_on_submit=True):
        question_column, ask_column = st.columns(
            [12, 1],
            vertical_alignment="bottom",
        )
        with question_column:
            question = st.text_input(
                "SolarIQ question",
                placeholder=(
                    "How much energy will Mecca produce on July 2, 2024?"
                ),
                label_visibility="collapsed",
                key="rag_question",
            )
        with ask_column:
            submitted = st.form_submit_button(
                "Ask",
                type="primary",
                width="stretch",
            )
        if submitted and question.strip():
            prompt = question.strip()

    suggestion_columns = st.columns(3)
    for index, (column, suggestion) in enumerate(
        zip(suggestion_columns, SUGGESTIONS)
    ):
        with column:
            if st.button(
                suggestion,
                width="stretch",
                key=f"rag_suggestion_{index}",
            ):
                prompt = suggestion

    if prompt:
        context = conversation_context(st.session_state.rag_messages)
        with st.spinner("Retrieving evidence and running your models…"):
            try:
                rag = initialize_rag(
                    get_secret("OLLAMA_API_KEY"),
                    get_secret("OLLAMA_HOST", "https://ollama.com"),
                    get_secret("OLLAMA_MODEL", "gpt-oss:20b"),
                )
                result = rag.process_query(prompt, context=context)
            except Exception as exc:
                result = {
                    "status": "error",
                    "error": (
                        "SolarIQ could not load the Python RAG pipeline. "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }

        st.session_state.rag_messages.append(
            {"role": "user", "content": prompt}
        )
        st.session_state.rag_messages.append(
            {"role": "assistant", "result": result}
        )

    for message in st.session_state.rag_messages:
        with st.chat_message(
            message["role"],
            avatar="🌞" if message["role"] == "assistant" else "👤",
        ):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_rag_result(message["result"])

    if st.session_state.rag_messages:
        if st.button(
            "Start a new conversation",
            width="stretch",
            key="new_rag_chat",
        ):
            st.session_state.rag_messages = []
            st.rerun()
