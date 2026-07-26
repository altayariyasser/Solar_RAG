"""Dedicated SolarIQ chatbot page using the full Llama-enabled RAG."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from rag import SolarRAG


SUGGESTIONS = [
    "How much solar energy will Riyadh produce today?",
    "Review Jeddah's solar conditions on 15 June 2024.",
    "What was the air-quality risk in Dammam on 10 March 2024?",
]

st.set_page_config(
    page_title="Ask SolarIQ",
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
      --text-1: #F4F6FB;
      --text-2: #B9C2D8;
      --text-3: #7D879C;
      --glass: rgba(13,18,34,.65);
      --glass-border: rgba(255,255,255,.09);
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
      color: var(--text-1);
      background:
        radial-gradient(circle at 86% 13%, rgba(253,184,19,.20), transparent 24%),
        radial-gradient(circle at 10% 88%, rgba(140,68,48,.28), transparent 30%),
        linear-gradient(155deg, #070B18 0%, #11162A 42%, #21131A 100%) !important;
      background-attachment: fixed !important;
    }

    [data-testid="stMainBlockContainer"] {
      width: min(1200px, 94vw);
      max-width: 1200px;
      padding: 24px 0 80px;
    }

    .chat-topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 38px;
    }

    .chat-brand {
      display: flex;
      align-items: center;
      gap: 11px;
    }

    .chat-brand-mark {
      width: 32px;
      height: 32px;
      border-radius: 10px;
      background: radial-gradient(circle at 32% 28%, #FFF3C8, var(--orange) 55%, var(--orange2));
      box-shadow: 0 0 24px rgba(253,184,19,.55);
    }

    .chat-brand strong {
      color: var(--text-1);
      font-family: 'Orbitron';
      font-size: 20px;
      font-weight: 900;
    }

    .chat-brand strong b { color: var(--orange); }

    .back-link {
      color: var(--text-2) !important;
      border: 1px solid var(--glass-border);
      border-radius: 100px;
      padding: 9px 15px;
      font-size: 12px;
      font-weight: 700;
      text-decoration: none !important;
    }

    .chat-hero {
      margin: 0 auto 28px;
      text-align: center;
    }

    .chat-hero .eyebrow {
      color: var(--orange);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 2.4px;
      text-transform: uppercase;
    }

    .chat-hero h1 {
      margin: 10px 0 12px;
      color: var(--text-1);
      font-family: 'Orbitron';
      font-size: clamp(30px, 5vw, 52px);
      font-weight: 900;
    }

    .chat-hero h1 span {
      color: transparent;
      background: linear-gradient(100deg, var(--yellow), var(--orange2));
      background-clip: text;
      -webkit-background-clip: text;
    }

    .chat-hero p {
      max-width: 720px;
      margin: 0 auto;
      color: var(--text-2);
      font-size: 14px;
      line-height: 1.75;
    }

    .st-key-chat_shell [data-testid="stVerticalBlockBorderWrapper"] {
      padding: 28px 30px;
      color: var(--text-1);
      background: var(--glass);
      border: 1px solid rgba(253,184,19,.30) !important;
      border-radius: 24px !important;
      box-shadow: 0 24px 60px -18px rgba(0,0,0,.55);
      backdrop-filter: blur(18px);
    }

    .st-key-chat_question [data-baseweb="input"] > div {
      min-height: 58px;
      color: var(--text-1);
      background: rgba(255,255,255,.055);
      border: 1px solid var(--glass-border);
      border-radius: 100px;
    }

    .st-key-chat_question input {
      color: var(--text-1) !important;
      padding: 0 22px !important;
    }

    .st-key-chat_question input::placeholder {
      color: var(--text-3) !important;
    }

    .st-key-chat_shell [data-testid="stFormSubmitButton"] button {
      min-height: 58px;
      color: #241500 !important;
      background: linear-gradient(135deg, var(--yellow), var(--orange2)) !important;
      border: 0 !important;
      border-radius: 100px !important;
      font-weight: 800 !important;
      box-shadow: 0 14px 34px -10px rgba(253,150,19,.65);
    }

    .st-key-suggestion_0 button,
    .st-key-suggestion_1 button,
    .st-key-suggestion_2 button,
    .st-key-clear_chat button {
      min-height: 40px;
      color: var(--text-2) !important;
      background: rgba(255,255,255,.045) !important;
      border: 1px solid var(--glass-border) !important;
      border-radius: 100px !important;
      font-size: 12px !important;
      font-weight: 700 !important;
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
      min-height: 108px;
      padding: 17px;
      background: rgba(255,255,255,.045);
      border: 1px solid var(--glass-border);
      border-radius: 15px;
    }

    [data-testid="stMetricLabel"] p {
      color: var(--text-3) !important;
      font-size: 10px !important;
      font-weight: 800 !important;
      text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
      color: var(--orange) !important;
      font-family: 'Orbitron' !important;
      font-size: 23px !important;
    }

    [data-testid="stExpander"],
    [data-testid="stAlert"] {
      color: var(--text-1);
      background: rgba(255,255,255,.035);
      border: 1px solid var(--glass-border) !important;
      border-radius: 14px !important;
    }

    .answer-meta {
      color: var(--text-3);
      font-size: 11px;
      font-weight: 700;
    }

    .llm-badge {
      display: inline-block;
      margin-top: 8px;
      padding: 5px 10px;
      color: var(--orange);
      background: rgba(253,184,19,.10);
      border: 1px solid rgba(253,184,19,.30);
      border-radius: 100px;
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 1px;
      text-transform: uppercase;
    }

    @media (max-width: 760px) {
      [data-testid="stMainBlockContainer"] { width: calc(100% - 24px); }
      .st-key-chat_shell [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 22px 17px;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


@st.cache_resource(show_spinner=False)
def initialize_rag(api_key: str, host: str, model: str) -> SolarRAG:
    system = SolarRAG(
        ollama_api_key=api_key,
        ollama_host=host,
        ollama_model=model,
    )
    system.setup()
    return system


def conversation_context(messages: list[Dict]) -> Dict:
    for message in reversed(messages):
        result = message.get("result")
        if result and result.get("status") == "success":
            return {
                "city": result.get("city"),
                "date": result.get("date"),
                "intents": result.get("intents"),
            }
    return {}


def show_result(result: Dict) -> None:
    if result.get("status") != "success":
        st.error(result.get("error", "SolarIQ could not answer that question."))
        return

    predictions = result["predictions"]
    st.markdown(result["llm_response"])
    st.markdown(
        (
            f"<p class='answer-meta'>{result['city']} · {result['date']} · "
            f"{result.get('source_label', 'Model inputs')}</p>"
        ),
        unsafe_allow_html=True,
    )
    solar, aqi, risk = st.columns(3)
    with solar:
        st.metric("Estimated solar energy", f"{predictions['solar_output_kwh']:.1f} kWh")
    with aqi:
        st.metric("Estimated AQI", f"{predictions['aqi_value']:.0f}")
    with risk:
        st.metric("Air-quality risk", predictions["aqi_risk_level"])

    with st.expander("Retrieved knowledge used in this answer"):
        for item in result["interpretations"]:
            st.markdown(f"- {item}")

    label = (
        "Llama explanation"
        if result.get("llm_status") == "ollama_cloud"
        else "Grounded local summary"
    )
    st.markdown(f"<span class='llm-badge'>{label}</span>", unsafe_allow_html=True)
    if result.get("llm_error"):
        st.caption("Llama was unavailable, so SolarIQ used its local grounded summary.")


st.markdown(
    """
    <div class="chat-topbar">
      <div class="chat-brand">
        <span class="chat-brand-mark"></span>
        <strong>Solar<b>IQ</b></strong>
      </div>
      <a class="back-link" href="/">← Dashboard</a>
    </div>
    <section class="chat-hero">
      <span class="eyebrow">Llama-powered solar analyst</span>
      <h1>Ask about the <span>solar day.</span></h1>
      <p>
        Ask about today or a historical date. SolarIQ retrieves the conditions,
        runs the solar and AQI models, retrieves relevant knowledge, and asks
        Llama for a grounded explanation.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("chatbot_messages", [])

with st.container(key="chat_shell", border=True):
    prompt = None
    with st.form("chatbot_form", clear_on_submit=True):
        question_column, ask_column = st.columns([12, 1], vertical_alignment="bottom")
        with question_column:
            question = st.text_input(
                "SolarIQ question",
                placeholder="How much energy did Mecca produce on July 2, 2024?",
                label_visibility="collapsed",
                key="chat_question",
            )
        with ask_column:
            submitted = st.form_submit_button("Ask", type="primary", width="stretch")
        if submitted and question.strip():
            prompt = question.strip()

    suggestion_columns = st.columns(3)
    for index, (column, suggestion) in enumerate(zip(suggestion_columns, SUGGESTIONS)):
        with column:
            if st.button(suggestion, width="stretch", key=f"suggestion_{index}"):
                prompt = suggestion

    if prompt:
        context = conversation_context(st.session_state.chatbot_messages)
        with st.spinner("Retrieving data, running models, and asking Llama…"):
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
                    "error": f"SolarIQ could not start: {type(exc).__name__}: {exc}",
                }
        st.session_state.chatbot_messages.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "result": result},
            ]
        )

    for message in st.session_state.chatbot_messages:
        with st.chat_message(
            message["role"],
            avatar="🌞" if message["role"] == "assistant" else "👤",
        ):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                show_result(message["result"])

    if st.session_state.chatbot_messages:
        if st.button("Start a new conversation", width="stretch", key="clear_chat"):
            st.session_state.chatbot_messages = []
            st.rerun()
