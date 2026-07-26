

Pasted text(6).txt
Document

streamlit_app(3).py
Python
change the theme of my streamlit the one named (streamlit_app.py) make it like this html exactly

Updated streamlit_app.py.

It now includes:

Animated sunset, clouds, dunes, and solar panels

Glass-style cards and golden controls

Matching Orbitron typography and hero section

City/date analysis panel

Responsive mobile design

Original RAG, models, charts, and chat logic preserved

Fully embedded styling—no separate theme file required

Run it with:

streamlit run streamlit_app.py

Library
/
streamlit_app.py


"""SolarIQ immersive dashboard and conversational solar analyst."""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta
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
    initial_sidebar_state="collapsed",
)

SOLARIQ_THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --orange: #FDB813;
  --orange2: #FF8A00;
  --yellow: #FFD54A;
  --ink: #0A0E1A;
  --ink2: #101628;
  --ink3: #182036;
  --text-1: #F4F6FB;
  --text-2: #B9C2D8;
  --text-3: #7D879C;
  --green: #38E27C;
  --red: #FF6B6B;
  --blue: #4FA0FF;
  --glass: rgba(13, 18, 34, 0.58);
  --glass-strong: rgba(13, 18, 34, 0.78);
  --glass-border: rgba(255, 255, 255, 0.09);
  --radius: 22px;
  --radius-sm: 14px;
  --shadow: 0 24px 60px -18px rgba(0, 0, 0, 0.55);
  --ease: cubic-bezier(.22, .9, .28, 1);
}

html, body, [class*="st-"], [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif;
}

html { scroll-behavior: smooth; }
body { background: var(--ink); }

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.stApp {
  color: var(--text-1);
  background: transparent !important;
}

[data-testid="stAppViewContainer"] {
  background:
    linear-gradient(180deg, rgba(7,11,24,.05), rgba(7,11,24,.28)),
    linear-gradient(180deg, #070B18 0%, #131A34 22%, #3A2350 46%, #8C4430 70%, #E0983A 92%, #F7C860 100%) !important;
  background-attachment: fixed !important;
}

[data-testid="stHeader"] {
  height: 0;
  background: transparent;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
  display: none !important;
}

[data-testid="stMainBlockContainer"] {
  width: min(1180px, 94vw);
  max-width: 1180px;
  padding: 18px 0 110px;
  position: relative;
  z-index: 2;
}

div[data-testid="stElementContainer"] {
  position: relative;
  z-index: 2;
}

div[data-testid="stElementContainer"]:has(.solariq-sky) {
  position: static;
  z-index: 0;
  height: 0;
}

::selection { background: rgba(253, 184, 19, .35); }

/* Living sky */
.solariq-sky {
  position: fixed;
  inset: 0;
  z-index: 0;
  overflow: hidden;
  pointer-events: none;
}

.sky-glow {
  position: absolute;
  right: 6%;
  top: 16%;
  width: 60vmin;
  height: 60vmin;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(253,184,19,.50), transparent 66%);
  filter: blur(8px);
}

.sun-wrap {
  position: absolute;
  right: 12%;
  top: 24%;
  width: 15vmin;
  height: 15vmin;
}

.sun-core {
  position: absolute;
  inset: 0;
  border-radius: 50%;
  background: radial-gradient(circle at 34% 30%, #FFF8E0, var(--yellow) 38%, var(--orange2) 78%);
  box-shadow: 0 0 9vmin 3vmin rgba(253,184,19,.5);
  animation: sun-breathe 6s ease-in-out infinite;
}

.sun-rays {
  position: absolute;
  inset: -46%;
  border-radius: 50%;
  background: repeating-conic-gradient(
    from 0deg,
    rgba(255,213,74,.14) 0 6deg,
    transparent 6deg 30deg
  );
  animation: sun-spin 70s linear infinite;
}

@keyframes sun-spin { to { transform: rotate(360deg); } }
@keyframes sun-breathe {
  0%, 100% { box-shadow: 0 0 9vmin 3vmin rgba(253,184,19,.5); }
  50% { box-shadow: 0 0 12vmin 4vmin rgba(253,184,19,.62); }
}

.clouds {
  position: absolute;
  inset: 0;
  opacity: .26;
}

.cloud {
  position: absolute;
  background: rgba(232,236,246,.80);
  border-radius: 999px;
  filter: blur(6px);
}

.cloud::before,
.cloud::after {
  content: "";
  position: absolute;
  background: inherit;
  border-radius: 50%;
}

.cloud-one {
  width: 22vmin;
  height: 6.5vmin;
  top: 16%;
  left: 6%;
  animation: cloud-drift 70s linear infinite;
}
.cloud-one::before { width: 11vmin; height: 11vmin; top: -5.5vmin; left: 3vmin; }
.cloud-one::after { width: 8vmin; height: 8vmin; top: -3.5vmin; left: 11vmin; }

.cloud-two {
  width: 17vmin;
  height: 5vmin;
  top: 34%;
  left: 52%;
  animation: cloud-drift 95s linear infinite -30s;
}
.cloud-two::before { width: 8.5vmin; height: 8.5vmin; top: -4vmin; left: 2.4vmin; }
.cloud-two::after { width: 6.5vmin; height: 6.5vmin; top: -2.6vmin; left: 8.5vmin; }

.cloud-three {
  width: 26vmin;
  height: 7vmin;
  top: 8%;
  left: 44%;
  animation: cloud-drift 120s linear infinite -60s;
}
.cloud-three::before { width: 13vmin; height: 13vmin; top: -6.5vmin; left: 4vmin; }
.cloud-three::after { width: 9.5vmin; height: 9.5vmin; top: -4.4vmin; left: 14vmin; }

@keyframes cloud-drift {
  from { transform: translateX(-30vw); }
  to { transform: translateX(130vw); }
}

.dust-particles {
  position: absolute;
  inset: 0;
  opacity: .15;
  background-image:
    radial-gradient(1.5px 1.5px at 12% 30%, rgba(255,214,160,.8), transparent 60%),
    radial-gradient(2px 2px at 34% 62%, rgba(255,200,140,.7), transparent 60%),
    radial-gradient(1.5px 1.5px at 58% 18%, rgba(255,220,170,.8), transparent 60%),
    radial-gradient(2px 2px at 72% 48%, rgba(255,205,150,.7), transparent 60%),
    radial-gradient(1.5px 1.5px at 88% 26%, rgba(255,215,165,.8), transparent 60%);
  background-size: 140% 140%;
  animation: dust-drift 9s linear infinite;
}

@keyframes dust-drift {
  from { background-position: 0 0; }
  to { background-position: -140px 40px; }
}

.heat-shimmer {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 18vh;
  height: 12vh;
  backdrop-filter: blur(.6px);
  opacity: .45;
  mask-image: linear-gradient(180deg, transparent, black 40%, transparent);
  animation: heat-shimmer 3.2s ease-in-out infinite alternate;
}

@keyframes heat-shimmer {
  from { transform: translateY(0) scaleY(1); }
  to { transform: translateY(-4px) scaleY(1.03); }
}

.horizon {
  position: absolute;
  left: -4%;
  right: -4%;
  bottom: 0;
  height: 34vh;
}

.dune {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
}

.dune-back {
  height: 78%;
  background: linear-gradient(180deg, transparent, #1A1428 62%, #12101F);
  clip-path: polygon(0 100%,0 58%,9% 42%,20% 60%,31% 32%,44% 55%,56% 25%,68% 52%,79% 36%,90% 58%,100% 44%,100% 100%);
  opacity: .85;
}

.dune-front {
  height: 52%;
  background: linear-gradient(180deg, transparent, #0D0B16 55%, #090812);
  clip-path: polygon(0 100%,0 66%,12% 50%,26% 68%,40% 44%,54% 64%,70% 40%,84% 62%,100% 50%,100% 100%);
}

.panel-field {
  position: absolute;
  left: 7%;
  right: 7%;
  bottom: 3.4vh;
  display: flex;
  gap: .8vw;
  height: 5.2vh;
  z-index: 2;
}

.panel-field i {
  flex: 1;
  transform: skewX(-8deg);
  background: linear-gradient(160deg, #22335C 0%, #0D1730 70%);
  border: 1px solid rgba(253,184,19,.35);
  border-radius: 4px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.12), 0 6px 14px -6px rgba(0,0,0,.6);
  position: relative;
  overflow: hidden;
}

.panel-field i::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(120deg, transparent 30%, rgba(253,184,19,.28) 50%, transparent 70%);
  transform: translateX(-120%);
  animation: panel-glint 5.5s ease-in-out infinite;
}

.panel-field i:nth-child(2n)::after { animation-delay: .6s; }
.panel-field i:nth-child(3n)::after { animation-delay: 1.2s; }

@keyframes panel-glint {
  0%, 60%, 100% { transform: translateX(-120%); }
  30% { transform: translateX(120%); }
}

/* Product shell */
.solariq-topbar {
  min-height: 62px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 0 4px 10px;
}

.solariq-brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.brand-mark {
  width: 30px;
  height: 30px;
  border-radius: 9px;
  background: radial-gradient(circle at 32% 28%, #FFF3C8, var(--orange) 55%, var(--orange2));
  box-shadow: 0 0 22px rgba(253,184,19,.55);
}

.brand-name {
  font-family: 'Orbitron';
  font-weight: 900;
  font-size: 21px;
  letter-spacing: .5px;
}

.brand-name b { color: var(--orange); }
.brand-sub {
  color: var(--text-3);
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: .4px;
}

.model-pulse {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-2);
  font-size: 12px;
  font-weight: 600;
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 100px;
  padding: 8px 14px;
  backdrop-filter: blur(10px);
}

.pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 10px var(--green);
  animation: model-pulse 2s ease-in-out infinite;
}

@keyframes model-pulse { 50% { opacity: .4; } }

.solariq-landing {
  max-width: 880px;
  margin: 4vh auto 26px;
  text-align: center;
  animation: fade-up .9s var(--ease) both;
}

.solariq-eyebrow {
  color: var(--orange);
  font-size: 12.5px;
  font-weight: 700;
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-bottom: 18px;
}

.solariq-headline {
  color: var(--text-1);
  font-family: 'Orbitron';
  font-size: clamp(32px, 5.6vw, 58px);
  font-weight: 900;
  line-height: 1.12;
  margin: 0;
}

.solariq-headline span {
  background: linear-gradient(100deg, var(--yellow), var(--orange2));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.solariq-sub {
  max-width: 650px;
  margin: 20px auto 0;
  color: var(--text-2);
  font-size: 15.5px;
  line-height: 1.75;
}

@keyframes fade-up {
  from { opacity: 0; transform: translateY(26px); }
  to { opacity: 1; transform: none; }
}

.st-key-launch_card [data-testid="stVerticalBlockBorderWrapper"] {
  max-width: 680px;
  margin: 0 auto;
  padding: 10px 12px 12px;
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 22px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
}

.st-key-launch_card [data-testid="stVerticalBlock"] {
  gap: 0;
}

label, [data-testid="stWidgetLabel"] p {
  color: var(--text-3) !important;
  font-size: 10.5px !important;
  font-weight: 700 !important;
  letter-spacing: 1.25px;
  text-transform: uppercase;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stDateInput"] input {
  color: var(--text-1) !important;
  background: rgba(255,255,255,.035) !important;
  border-color: var(--glass-border) !important;
  border-radius: 14px !important;
}

[data-baseweb="select"] > div:hover,
[data-baseweb="input"] > div:hover {
  border-color: rgba(253,184,19,.50) !important;
}

[data-baseweb="popover"],
[role="listbox"],
[data-baseweb="menu"] {
  color: var(--text-1) !important;
  background: var(--ink2) !important;
}

.stButton > button,
.stDownloadButton > button {
  min-height: 46px;
  color: var(--text-1);
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 14px;
  font-weight: 700;
  transition: transform .18s var(--ease), border-color .25s, box-shadow .25s;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
  color: var(--text-1);
  border-color: rgba(253,184,19,.50);
  transform: translateY(-2px);
}

.stButton > button[kind="primary"],
.st-key-launch_analysis .stButton > button {
  color: #241500 !important;
  background: linear-gradient(135deg, var(--yellow), var(--orange2)) !important;
  border: 0 !important;
  box-shadow: 0 14px 34px -10px rgba(253,150,19,.65);
}

.st-key-launch_analysis .stButton > button {
  min-height: 66px;
  margin-top: 15px;
  font-size: 15px;
}

.solariq-trust-row {
  display: flex;
  justify-content: center;
  gap: clamp(20px, 4vw, 48px);
  margin: 34px 0 48px;
  flex-wrap: wrap;
  text-align: center;
}

.solariq-trust {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.solariq-trust b {
  color: var(--orange);
  font-family: 'Orbitron';
  font-size: 22px;
}

.solariq-trust span {
  color: var(--text-3);
  font-size: 11.5px;
  font-weight: 600;
}

/* Streamlit cards translated from the reference HTML */
h1, h2, h3 {
  color: var(--text-1) !important;
}

h1 {
  font-family: 'Orbitron' !important;
  font-weight: 900 !important;
}

h2, h3 {
  font-weight: 800 !important;
}

p, li, span {
  text-rendering: optimizeLegibility;
}

[data-testid="stCaptionContainer"] p,
.stCaption {
  color: var(--text-3) !important;
}

div[data-testid="stMetric"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"],
[data-testid="stChatMessage"] {
  background: var(--glass);
  border: 1px solid var(--glass-border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
}

div[data-testid="stMetric"] {
  min-height: 138px;
  padding: 22px 24px;
  transition: transform .35s var(--ease), border-color .35s;
}

div[data-testid="stMetric"]:hover,
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  transform: translateY(-4px);
  border-color: rgba(253,184,19,.32) !important;
}

[data-testid="stMetricLabel"] p {
  color: var(--text-2) !important;
  font-size: 12px !important;
  font-weight: 800 !important;
  letter-spacing: 1.1px;
  text-transform: uppercase;
}

[data-testid="stMetricValue"] {
  color: var(--orange) !important;
  font-family: 'Orbitron' !important;
  font-size: clamp(24px, 3vw, 38px) !important;
  font-weight: 900 !important;
}

[data-testid="stMetricDelta"] {
  color: var(--text-3) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
  padding: 22px;
}

[data-testid="stExpander"] {
  overflow: hidden;
}

[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
  color: var(--text-1) !important;
  background: transparent !important;
}

[data-testid="stChatMessage"] {
  padding: 18px 20px;
  margin-bottom: 12px;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: rgba(253,184,19,.09);
  border-color: rgba(253,184,19,.28) !important;
}

[data-testid="stChatInput"] {
  background: var(--glass-strong) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 100px !important;
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
}

[data-testid="stChatInput"] textarea {
  color: var(--text-1) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
  color: var(--text-3) !important;
}

[data-testid="stChatInputSubmitButton"] {
  color: #241500 !important;
  background: linear-gradient(135deg, var(--yellow), var(--orange2)) !important;
  border-radius: 50% !important;
}

[data-testid="stPills"] button {
  color: var(--text-2) !important;
  background: rgba(255,255,255,.045) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 100px !important;
}

[data-testid="stPills"] button:hover,
[data-testid="stPills"] button[aria-pressed="true"] {
  color: var(--text-1) !important;
  border-color: rgba(253,184,19,.45) !important;
  background: rgba(253,184,19,.12) !important;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
  overflow: hidden;
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-sm);
}

[data-testid="stAlert"] {
  color: var(--text-1);
  background: var(--glass-strong);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-sm);
}

.solariq-section-head {
  margin: 22px 4px 16px;
}

.solariq-section-head h2 {
  margin: 0;
  font-family: 'Orbitron';
  font-size: clamp(21px, 3vw, 30px);
  font-weight: 900;
}

.solariq-section-head p {
  color: var(--text-3);
  font-size: 13px;
  font-weight: 600;
  margin-top: 6px;
}

.solariq-footer {
  color: var(--text-3);
  font-size: 11px;
  margin-top: 34px;
  text-align: center;
}

@media (max-width: 760px) {
  [data-testid="stMainBlockContainer"] {
    width: min(94vw, 1180px);
    padding-top: 10px;
  }
  .solariq-topbar { align-items: flex-start; }
  .brand-sub { display: none; }
  .model-pulse { font-size: 10px; padding: 7px 10px; }
  .solariq-landing { margin-top: 2vh; }
  .solariq-sub { font-size: 14px; }
  .st-key-launch_analysis .stButton > button {
    min-height: 48px;
    margin-top: 4px;
  }
  .solariq-trust-row { margin-bottom: 34px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .2s !important;
  }
}
</style>

<div class="solariq-sky" aria-hidden="true">
  <div class="sky-glow"></div>
  <div class="sun-wrap">
    <div class="sun-rays"></div>
    <div class="sun-core"></div>
  </div>
  <div class="clouds">
    <div class="cloud cloud-one"></div>
    <div class="cloud cloud-two"></div>
    <div class="cloud cloud-three"></div>
  </div>
  <div class="dust-particles"></div>
  <div class="heat-shimmer"></div>
  <div class="horizon">
    <div class="dune dune-back"></div>
    <div class="dune dune-front"></div>
    <div class="panel-field">
      <i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i>
    </div>
  </div>
</div>
"""
st.html(SOLARIQ_THEME)


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


st.html(
    """
    <header class="solariq-topbar">
      <div class="solariq-brand">
        <span class="brand-mark" aria-hidden="true"></span>
        <span class="brand-name">Solar<b>IQ</b></span>
        <span class="brand-sub">Solar Intelligence · KSA</span>
      </div>
      <div class="model-pulse">
        <span class="pulse-dot"></span>
        2 analytical models · RAG ready
      </div>
    </header>

    <section class="solariq-landing">
      <p class="solariq-eyebrow">AI-powered daily forecasting</p>
      <h1 class="solariq-headline">
        Know your solar day<br>
        <span>before the sun does.</span>
      </h1>
      <p class="solariq-sub">
        Pick a city and a date. SolarIQ reads weather and air quality,
        runs the trained models, and returns solar potential,
        environmental risk, and grounded recommendations in seconds.
      </p>
    </section>
    """
)

launch_prompt = None
today = datetime.now().date()
with st.container(key="launch_card", border=True):
    city_column, date_column, action_column = st.columns(
        [1.15, 1.15, 0.8],
        vertical_alignment="bottom",
    )
    with city_column:
        selected_market = st.selectbox(
            "City",
            rag.data_loader.cities,
            index=rag.data_loader.cities.index("Riyadh"),
            key="market_selector",
        )
    with date_column:
        selected_date = st.date_input(
            "Date",
            value=today,
            min_value=datetime(2015, 1, 1).date(),
            max_value=today + timedelta(days=7),
            key="analysis_date",
        )
    with action_column:
        if st.button(
            "Analyze  →",
            type="primary",
            width="stretch",
            key="launch_analysis",
        ):
            launch_prompt = (
                f"Analyze solar potential, weather conditions, air-quality risk, "
                f"and site suitability in {selected_market} on "
                f"{selected_date.strftime('%B %d, %Y')}."
            )


frame = rag.data_loader.df_merged
historical_days = int(frame["Date"].nunique()) if frame is not None else 0
st.html(
    f"""
    <div class="solariq-trust-row">
      <div class="solariq-trust">
        <b>{len(rag.data_loader.cities)}</b>
        <span>Saudi cities</span>
      </div>
      <div class="solariq-trust">
        <b>{historical_days}</b>
        <span>historical days</span>
      </div>
      <div class="solariq-trust">
        <b>2</b>
        <span>analytical models</span>
      </div>
      <div class="solariq-trust">
        <b>RAG</b>
        <span>grounded guidance</span>
      </div>
    </div>
    """
)

st.html(
    """
    <div class="solariq-section-head">
      <h2>Project intelligence</h2>
      <p>Portfolio coverage, historical depth, and analytical capability</p>
    </div>
    """
)
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
            color="#FDB813",
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


st.html(
    """
    <div class="solariq-section-head">
      <h2>Ask SolarIQ</h2>
      <p>
        parse → retrieve → predict → ground · Ask naturally about a market
        and date; short follow-ups retain the previous context.
      </p>
    </div>
    """
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
if launch_prompt:
    prompt = launch_prompt
elif selected_suggestion:
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


if st.session_state.messages:
    if st.button(
        "Analyze another day",
        icon=":material/refresh:",
        width="stretch",
        key="new_analysis",
    ):
        st.session_state.messages = []
        st.rerun()

st.html(
    f"""
    <p class="solariq-footer">
      SolarIQ · trained on merged KSA weather, air-quality, and solar-generation
      data · {APP_BUILD}
    </p>
    """
)
Library
/
streamlit_app.py


"""SolarIQ immersive dashboard and conversational solar analyst."""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta
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
    initial_sidebar_state="collapsed",
)

SOLARIQ_THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --orange: #FDB813;
  --orange2: #FF8A00;
  --yellow: #FFD54A;
  --ink: #0A0E1A;
  --ink2: #101628;
  --ink3: #182036;
  --text-1: #F4F6FB;
  --text-2: #B9C2D8;
  --text-3: #7D879C;
  --green: #38E27C;
  --red: #FF6B6B;
  --blue: #4FA0FF;
  --glass: rgba(13, 18, 34, 0.58);
  --glass-strong: rgba(13, 18, 34, 0.78);
  --glass-border: rgba(255, 255, 255, 0.09);
  --radius: 22px;
  --radius-sm: 14px;
  --shadow: 0 24px 60px -18px rgba(0, 0, 0, 0.55);
  --ease: cubic-bezier(.22, .9, .28, 1);
}

html, body, [class*="st-"], [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif;
}

html { scroll-behavior: smooth; }
body { background: var(--ink); }

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.stApp {
  color: var(--text-1);
  background: transparent !important;
}

[data-testid="stAppViewContainer"] {
  background:
    linear-gradient(180deg, rgba(7,11,24,.05), rgba(7,11,24,.28)),
    linear-gradient(180deg, #070B18 0%, #131A34 22%, #3A2350 46%, #8C4430 70%, #E0983A 92%, #F7C860 100%) !important;
  background-attachment: fixed !important;
}

[data-testid="stHeader"] {
  height: 0;
  background: transparent;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
  display: none !important;
}

[data-testid="stMainBlockContainer"] {
  width: min(1180px, 94vw);
  max-width: 1180px;
  padding: 18px 0 110px;
  position: relative;
  z-index: 2;
}

div[data-testid="stElementContainer"] {
  position: relative;
  z-index: 2;
}

div[data-testid="stElementContainer"]:has(.solariq-sky) {
  position: static;
  z-index: 0;
  height: 0;
}

::selection { background: rgba(253, 184, 19, .35); }

/* Living sky */
.solariq-sky {
  position: fixed;
  inset: 0;
  z-index: 0;
  overflow: hidden;
  pointer-events: none;
}

.sky-glow {
  position: absolute;
  right: 6%;
  top: 16%;
  width: 60vmin;
  height: 60vmin;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(253,184,19,.50), transparent 66%);
  filter: blur(8px);
}

.sun-wrap {
  position: absolute;
  right: 12%;
  top: 24%;
  width: 15vmin;
  height: 15vmin;
}

.sun-core {
  position: absolute;
  inset: 0;
  border-radius: 50%;
  background: radial-gradient(circle at 34% 30%, #FFF8E0, var(--yellow) 38%, var(--orange2) 78%);
  box-shadow: 0 0 9vmin 3vmin rgba(253,184,19,.5);
  animation: sun-breathe 6s ease-in-out infinite;
}

.sun-rays {
  position: absolute;
  inset: -46%;
  border-radius: 50%;
  background: repeating-conic-gradient(
    from 0deg,
    rgba(255,213,74,.14) 0 6deg,
    transparent 6deg 30deg
  );
  animation: sun-spin 70s linear infinite;
}

@keyframes sun-spin { to { transform: rotate(360deg); } }
@keyframes sun-breathe {
  0%, 100% { box-shadow: 0 0 9vmin 3vmin rgba(253,184,19,.5); }
  50% { box-shadow: 0 0 12vmin 4vmin rgba(253,184,19,.62); }
}

.clouds {
  position: absolute;
  inset: 0;
  opacity: .26;
}

.cloud {
  position: absolute;
  background: rgba(232,236,246,.80);
  border-radius: 999px;
  filter: blur(6px);
}

.cloud::before,
.cloud::after {
  content: "";
  position: absolute;
  background: inherit;
  border-radius: 50%;
}

.cloud-one {
  width: 22vmin;
  height: 6.5vmin;
  top: 16%;
  left: 6%;
  animation: cloud-drift 70s linear infinite;
}
.cloud-one::before { width: 11vmin; height: 11vmin; top: -5.5vmin; left: 3vmin; }
.cloud-one::after { width: 8vmin; height: 8vmin; top: -3.5vmin; left: 11vmin; }

.cloud-two {
  width: 17vmin;
  height: 5vmin;
  top: 34%;
  left: 52%;
  animation: cloud-drift 95s linear infinite -30s;
}
.cloud-two::before { width: 8.5vmin; height: 8.5vmin; top: -4vmin; left: 2.4vmin; }
.cloud-two::after { width: 6.5vmin; height: 6.5vmin; top: -2.6vmin; left: 8.5vmin; }

.cloud-three {
  width: 26vmin;
  height: 7vmin;
  top: 8%;
  left: 44%;
  animation: cloud-drift 120s linear infinite -60s;
}
.cloud-three::before { width: 13vmin; height: 13vmin; top: -6.5vmin; left: 4vmin; }
.cloud-three::after { width: 9.5vmin; height: 9.5vmin; top: -4.4vmin; left: 14vmin; }

@keyframes cloud-drift {
  from { transform: translateX(-30vw); }
  to { transform: translateX(130vw); }
}

.dust-particles {
  position: absolute;
  inset: 0;
  opacity: .15;
  background-image:
    radial-gradient(1.5px 1.5px at 12% 30%, rgba(255,214,160,.8), transparent 60%),
    radial-gradient(2px 2px at 34% 62%, rgba(255,200,140,.7), transparent 60%),
    radial-gradient(1.5px 1.5px at 58% 18%, rgba(255,220,170,.8), transparent 60%),
    radial-gradient(2px 2px at 72% 48%, rgba(255,205,150,.7), transparent 60%),
    radial-gradient(1.5px 1.5px at 88% 26%, rgba(255,215,165,.8), transparent 60%);
  background-size: 140% 140%;
  animation: dust-drift 9s linear infinite;
}

@keyframes dust-drift {
  from { background-position: 0 0; }
  to { background-position: -140px 40px; }
}

.heat-shimmer {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 18vh;
  height: 12vh;
  backdrop-filter: blur(.6px);
  opacity: .45;
  mask-image: linear-gradient(180deg, transparent, black 40%, transparent);
  animation: heat-shimmer 3.2s ease-in-out infinite alternate;
}

@keyframes heat-shimmer {
  from { transform: translateY(0) scaleY(1); }
  to { transform: translateY(-4px) scaleY(1.03); }
}

.horizon {
  position: absolute;
  left: -4%;
  right: -4%;
  bottom: 0;
  height: 34vh;
}

.dune {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
}

.dune-back {
  height: 78%;
  background: linear-gradient(180deg, transparent, #1A1428 62%, #12101F);
  clip-path: polygon(0 100%,0 58%,9% 42%,20% 60%,31% 32%,44% 55%,56% 25%,68% 52%,79% 36%,90% 58%,100% 44%,100% 100%);
  opacity: .85;
}

.dune-front {
  height: 52%;
  background: linear-gradient(180deg, transparent, #0D0B16 55%, #090812);
  clip-path: polygon(0 100%,0 66%,12% 50%,26% 68%,40% 44%,54% 64%,70% 40%,84% 62%,100% 50%,100% 100%);
}

.panel-field {
  position: absolute;
  left: 7%;
  right: 7%;
  bottom: 3.4vh;
  display: flex;
  gap: .8vw;
  height: 5.2vh;
  z-index: 2;
}

.panel-field i {
  flex: 1;
  transform: skewX(-8deg);
  background: linear-gradient(160deg, #22335C 0%, #0D1730 70%);
  border: 1px solid rgba(253,184,19,.35);
  border-radius: 4px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.12), 0 6px 14px -6px rgba(0,0,0,.6);
  position: relative;
  overflow: hidden;
}

.panel-field i::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(120deg, transparent 30%, rgba(253,184,19,.28) 50%, transparent 70%);
  transform: translateX(-120%);
  animation: panel-glint 5.5s ease-in-out infinite;
}

.panel-field i:nth-child(2n)::after { animation-delay: .6s; }
.panel-field i:nth-child(3n)::after { animation-delay: 1.2s; }

@keyframes panel-glint {
  0%, 60%, 100% { transform: translateX(-120%); }
  30% { transform: translateX(120%); }
}

/* Product shell */
.solariq-topbar {
  min-height: 62px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 0 4px 10px;
}

.solariq-brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.brand-mark {
  width: 30px;
  height: 30px;
  border-radius: 9px;
  background: radial-gradient(circle at 32% 28%, #FFF3C8, var(--orange) 55%, var(--orange2));
  box-shadow: 0 0 22px rgba(253,184,19,.55);
}

.brand-name {
  font-family: 'Orbitron';
  font-weight: 900;
  font-size: 21px;
  letter-spacing: .5px;
}

.brand-name b { color: var(--orange); }
.brand-sub {
  color: var(--text-3);
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: .4px;
}

.model-pulse {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-2);
  font-size: 12px;
  font-weight: 600;
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 100px;
  padding: 8px 14px;
  backdrop-filter: blur(10px);
}

.pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 10px var(--green);
  animation: model-pulse 2s ease-in-out infinite;
}

@keyframes model-pulse { 50% { opacity: .4; } }

.solariq-landing {
  max-width: 880px;
  margin: 4vh auto 26px;
  text-align: center;
  animation: fade-up .9s var(--ease) both;
}

.solariq-eyebrow {
  color: var(--orange);
  font-size: 12.5px;
  font-weight: 700;
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-bottom: 18px;
}

.solariq-headline {
  color: var(--text-1);
  font-family: 'Orbitron';
  font-size: clamp(32px, 5.6vw, 58px);
  font-weight: 900;
  line-height: 1.12;
  margin: 0;
}

.solariq-headline span {
  background: linear-gradient(100deg, var(--yellow), var(--orange2));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.solariq-sub {
  max-width: 650px;
  margin: 20px auto 0;
  color: var(--text-2);
  font-size: 15.5px;
  line-height: 1.75;
}

@keyframes fade-up {
  from { opacity: 0; transform: translateY(26px); }
  to { opacity: 1; transform: none; }
}

.st-key-launch_card [data-testid="stVerticalBlockBorderWrapper"] {
  max-width: 680px;
  margin: 0 auto;
  padding: 10px 12px 12px;
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 22px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
}

.st-key-launch_card [data-testid="stVerticalBlock"] {
  gap: 0;
}

label, [data-testid="stWidgetLabel"] p {
  color: var(--text-3) !important;
  font-size: 10.5px !important;
  font-weight: 700 !important;
  letter-spacing: 1.25px;
  text-transform: uppercase;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stDateInput"] input {
  color: var(--text-1) !important;
  background: rgba(255,255,255,.035) !important;
  border-color: var(--glass-border) !important;
  border-radius: 14px !important;
}

[data-baseweb="select"] > div:hover,
[data-baseweb="input"] > div:hover {
  border-color: rgba(253,184,19,.50) !important;
}

[data-baseweb="popover"],
[role="listbox"],
[data-baseweb="menu"] {
  color: var(--text-1) !important;
  background: var(--ink2) !important;
}

.stButton > button,
.stDownloadButton > button {
  min-height: 46px;
  color: var(--text-1);
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 14px;
  font-weight: 700;
  transition: transform .18s var(--ease), border-color .25s, box-shadow .25s;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
  color: var(--text-1);
  border-color: rgba(253,184,19,.50);
  transform: translateY(-2px);
}

.stButton > button[kind="primary"],
.st-key-launch_analysis .stButton > button {
  color: #241500 !important;
  background: linear-gradient(135deg, var(--yellow), var(--orange2)) !important;
  border: 0 !important;
  box-shadow: 0 14px 34px -10px rgba(253,150,19,.65);
}

.st-key-launch_analysis .stButton > button {
  min-height: 66px;
  margin-top: 15px;
  font-size: 15px;
}

.solariq-trust-row {
  display: flex;
  justify-content: center;
  gap: clamp(20px, 4vw, 48px);
  margin: 34px 0 48px;
  flex-wrap: wrap;
  text-align: center;
}

.solariq-trust {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.solariq-trust b {
  color: var(--orange);
  font-family: 'Orbitron';
  font-size: 22px;
}

.solariq-trust span {
  color: var(--text-3);
  font-size: 11.5px;
  font-weight: 600;
}

/* Streamlit cards translated from the reference HTML */
h1, h2, h3 {
  color: var(--text-1) !important;
}

h1 {
  font-family: 'Orbitron' !important;
  font-weight: 900 !important;
}

h2, h3 {
  font-weight: 800 !important;
}

p, li, span {
  text-rendering: optimizeLegibility;
}

[data-testid="stCaptionContainer"] p,
.stCaption {
  color: var(--text-3) !important;
}

div[data-testid="stMetric"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"],
[data-testid="stChatMessage"] {
  background: var(--glass);
  border: 1px solid var(--glass-border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
}

div[data-testid="stMetric"] {
  min-height: 138px;
  padding: 22px 24px;
  transition: transform .35s var(--ease), border-color .35s;
}

div[data-testid="stMetric"]:hover,
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  transform: translateY(-4px);
  border-color: rgba(253,184,19,.32) !important;
}

[data-testid="stMetricLabel"] p {
  color: var(--text-2) !important;
  font-size: 12px !important;
  font-weight: 800 !important;
  letter-spacing: 1.1px;
  text-transform: uppercase;
}

[data-testid="stMetricValue"] {
  color: var(--orange) !important;
  font-family: 'Orbitron' !important;
  font-size: clamp(24px, 3vw, 38px) !important;
  font-weight: 900 !important;
}

[data-testid="stMetricDelta"] {
  color: var(--text-3) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
  padding: 22px;
}

[data-testid="stExpander"] {
  overflow: hidden;
}

[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
  color: var(--text-1) !important;
  background: transparent !important;
}

[data-testid="stChatMessage"] {
  padding: 18px 20px;
  margin-bottom: 12px;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: rgba(253,184,19,.09);
  border-color: rgba(253,184,19,.28) !important;
}

[data-testid="stChatInput"] {
  background: var(--glass-strong) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 100px !important;
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
}

[data-testid="stChatInput"] textarea {
  color: var(--text-1) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
  color: var(--text-3) !important;
}

[data-testid="stChatInputSubmitButton"] {
  color: #241500 !important;
  background: linear-gradient(135deg, var(--yellow), var(--orange2)) !important;
  border-radius: 50% !important;
}

[data-testid="stPills"] button {
  color: var(--text-2) !important;
  background: rgba(255,255,255,.045) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 100px !important;
}

[data-testid="stPills"] button:hover,
[data-testid="stPills"] button[aria-pressed="true"] {
  color: var(--text-1) !important;
  border-color: rgba(253,184,19,.45) !important;
  background: rgba(253,184,19,.12) !important;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
  overflow: hidden;
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-sm);
}

[data-testid="stAlert"] {
  color: var(--text-1);
  background: var(--glass-strong);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-sm);
}

.solariq-section-head {
  margin: 22px 4px 16px;
}

.solariq-section-head h2 {
  margin: 0;
  font-family: 'Orbitron';
  font-size: clamp(21px, 3vw, 30px);
  font-weight: 900;
}

.solariq-section-head p {
  color: var(--text-3);
  font-size: 13px;
  font-weight: 600;
  margin-top: 6px;
}

.solariq-footer {
  color: var(--text-3);
  font-size: 11px;
  margin-top: 34px;
  text-align: center;
}

@media (max-width: 760px) {
  [data-testid="stMainBlockContainer"] {
    width: min(94vw, 1180px);
    padding-top: 10px;
  }
  .solariq-topbar { align-items: flex-start; }
  .brand-sub { display: none; }
  .model-pulse { font-size: 10px; padding: 7px 10px; }
  .solariq-landing { margin-top: 2vh; }
  .solariq-sub { font-size: 14px; }
  .st-key-launch_analysis .stButton > button {
    min-height: 48px;
    margin-top: 4px;
  }
  .solariq-trust-row { margin-bottom: 34px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .2s !important;
  }
}
</style>

<div class="solariq-sky" aria-hidden="true">
  <div class="sky-glow"></div>
  <div class="sun-wrap">
    <div class="sun-rays"></div>
    <div class="sun-core"></div>
  </div>
  <div class="clouds">
    <div class="cloud cloud-one"></div>
    <div class="cloud cloud-two"></div>
    <div class="cloud cloud-three"></div>
  </div>
  <div class="dust-particles"></div>
  <div class="heat-shimmer"></div>
  <div class="horizon">
    <div class="dune dune-back"></div>
    <div class="dune dune-front"></div>
    <div class="panel-field">
      <i></i><i></i><i></i><i></i><i></i><i></i>
      <i></i><i></i><i></i><i></i><i></i><i></i>
    </div>
  </div>
</div>
"""
st.html(SOLARIQ_THEME)


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


st.html(
    """
    <header class="solariq-topbar">
      <div class="solariq-brand">
        <span class="brand-mark" aria-hidden="true"></span>
        <span class="brand-name">Solar<b>IQ</b></span>
        <span class="brand-sub">Solar Intelligence · KSA</span>
      </div>
      <div class="model-pulse">
        <span class="pulse-dot"></span>
        2 analytical models · RAG ready
      </div>
    </header>

    <section class="solariq-landing">
      <p class="solariq-eyebrow">AI-powered daily forecasting</p>
      <h1 class="solariq-headline">
        Know your solar day<br>
        <span>before the sun does.</span>
      </h1>
      <p class="solariq-sub">
        Pick a city and a date. SolarIQ reads weather and air quality,
        runs the trained models, and returns solar potential,
        environmental risk, and grounded recommendations in seconds.
      </p>
    </section>
    """
)

launch_prompt = None
today = datetime.now().date()
with st.container(key="launch_card", border=True):
    city_column, date_column, action_column = st.columns(
        [1.15, 1.15, 0.8],
        vertical_alignment="bottom",
    )
    with city_column:
        selected_market = st.selectbox(
            "City",
            rag.data_loader.cities,
            index=rag.data_loader.cities.index("Riyadh"),
            key="market_selector",
        )
    with date_column:
        selected_date = st.date_input(
            "Date",
            value=today,
            min_value=datetime(2015, 1, 1).date(),
            max_value=today + timedelta(days=7),
            key="analysis_date",
        )
    with action_column:
        if st.button(
            "Analyze  →",
            type="primary",
            width="stretch",
            key="launch_analysis",
        ):
            launch_prompt = (
                f"Analyze solar potential, weather conditions, air-quality risk, "
                f"and site suitability in {selected_market} on "
                f"{selected_date.strftime('%B %d, %Y')}."
            )


frame = rag.data_loader.df_merged
historical_days = int(frame["Date"].nunique()) if frame is not None else 0
st.html(
    f"""
    <div class="solariq-trust-row">
      <div class="solariq-trust">
        <b>{len(rag.data_loader.cities)}</b>
        <span>Saudi cities</span>
      </div>
      <div class="solariq-trust">
        <b>{historical_days}</b>
        <span>historical days</span>
      </div>
      <div class="solariq-trust">
        <b>2</b>
        <span>analytical models</span>
      </div>
      <div class="solariq-trust">
        <b>RAG</b>
        <span>grounded guidance</span>
      </div>
    </div>
    """
)

st.html(
    """
    <div class="solariq-section-head">
      <h2>Project intelligence</h2>
      <p>Portfolio coverage, historical depth, and analytical capability</p>
    </div>
    """
)
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
            color="#FDB813",
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


st.html(
    """
    <div class="solariq-section-head">
      <h2>Ask SolarIQ</h2>
      <p>
        parse → retrieve → predict → ground · Ask naturally about a market
        and date; short follow-ups retain the previous context.
      </p>
    </div>
    """
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
if launch_prompt:
    prompt = launch_prompt
elif selected_suggestion:
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


if st.session_state.messages:
    if st.button(
        "Analyze another day",
        icon=":material/refresh:",
        width="stretch",
        key="new_analysis",
    ):
        st.session_state.messages = []
        st.rerun()

st.html(
    f"""
    <p class="solariq-footer">
      SolarIQ · trained on merged KSA weather, air-quality, and solar-generation
      data · {APP_BUILD}
    </p>
    """
)
