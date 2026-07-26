"""SolarIQ dashboard host with a separate Llama chatbot page."""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


APP_DIR = Path(__file__).resolve().parent
HTML_FILE = APP_DIR / "solariq_standalone.html"

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
      --text-1: #F4F6FB;
      --text-2: #B9C2D8;
      --text-3: #7D879C;
      --glass: rgba(13, 18, 34, 0.62);
      --glass-border: rgba(255, 255, 255, 0.09);
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
        radial-gradient(circle at 92% 8%, rgba(255,138,0,.14), transparent 28%),
        linear-gradient(180deg, #0A0E1A 0%, #14101D 55%, #24151B 100%) !important;
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

    .chat-fab {
      position: fixed;
      top: 22px;
      right: 24px;
      z-index: 2147483647;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 14px 22px;
      color: #241500;
      background: linear-gradient(135deg, var(--yellow), var(--orange2));
      border: 1px solid rgba(255, 213, 74, .85);
      border-radius: 999px;
      box-shadow:
        0 16px 38px -9px rgba(253,150,19,.80),
        0 0 0 5px rgba(253,184,19,.10);
      font: 800 14px 'Inter', sans-serif;
      text-decoration: none;
      transition: transform .2s ease, box-shadow .2s ease;
      backdrop-filter: blur(14px);
    }

    .chat-fab:hover {
      color: #241500;
      transform: translateY(-2px) scale(1.02);
      box-shadow:
        0 20px 44px -8px rgba(253,150,19,.92),
        0 0 0 6px rgba(253,184,19,.13);
    }

    @media (max-width: 760px) {
      .chat-fab {
        top: auto;
        right: 14px;
        bottom: 16px;
        padding: 13px 18px;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_interface_html(path: str, modified_ns: int) -> str:
    """Cache the large dashboard source until the file changes."""
    del modified_ns
    return Path(path).read_text(encoding="utf-8")


if not HTML_FILE.exists():
    st.error("The SolarIQ interface file `solariq_standalone.html` is missing.")
    st.stop()

st.markdown(
    """
    <a
      class="chat-fab"
      href="/chatbot"
      target="_blank"
      rel="noopener noreferrer"
      aria-label="Open the SolarIQ chatbot in a new tab"
    >
      <span aria-hidden="true">☀</span>
      Open SolarIQ Chatbot ↗
    </a>
    """,
    unsafe_allow_html=True,
)

components.html(
    load_interface_html(str(HTML_FILE), HTML_FILE.stat().st_mtime_ns),
    height=1420,
    scrolling=True,
)
