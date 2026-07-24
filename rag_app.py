"""
Interactive Streamlit Web Interface for Solar Project RAG System
Provides a user-friendly UI for querying weather, air quality, and solar predictions
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from rag import SolarRAG
import sys
from datetime import datetime, timedelta

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="☀️ Solar Project RAG System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1em;
        padding: 10px 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .good {
        color: #31a049;
        font-weight: bold;
    }
    .moderate {
        color: #ff9800;
        font-weight: bold;
    }
    .unhealthy {
        color: #d32f2f;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE & INITIALIZATION
# ============================================================================
@st.cache_resource
def initialize_rag():
    """Initialize RAG system once per session"""
    rag = SolarRAG()
    if rag.setup():
        return rag
    else:
        st.error("Failed to initialize RAG system")
        st.stop()

rag = initialize_rag()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_aqi_color(aqi_value):
    """Return color based on AQI value"""
    if aqi_value < 50:
        return "#31a049"  # Green
    elif aqi_value < 100:
        return "#ff9800"  # Orange
    else:
        return "#d32f2f"  # Red

def get_risk_color(risk_level):
    """Return color based on risk level"""
    colors = {
        "Good": "#31a049",
        "Moderate": "#ff9800",
        "Unhealthy": "#d32f2f"
    }
    return colors.get(risk_level, "#757575")

def format_prediction(predictions):
    """Format predictions for display"""
    formatted = {}
    if 'solar_output_kwh' in predictions:
        formatted['Solar Output (kWh)'] = f"{predictions['solar_output_kwh']:.2f}"
    if 'aqi_value' in predictions:
        formatted['AQI Value'] = f"{predictions['aqi_value']:.0f}"
    if 'aqi_risk_level' in predictions:
        formatted['Air Quality Risk'] = predictions['aqi_risk_level']
    return formatted

# ============================================================================
# MAIN INTERFACE
# ============================================================================
# Header
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://img.icons8.com/color/96/000000/sun.png", width=80)
with col2:
    st.title("☀️ Solar Project RAG System")
    st.markdown("*Predict solar output & air quality for Saudi Arabia*")

st.divider()

# ============================================================================
# SIDEBAR - QUERY INPUT
# ============================================================================
with st.sidebar:
    st.header("🔍 Query Input")
    
    # Query Mode Selection
    query_mode = st.radio(
        "How would you like to query?",
        ["Natural Language", "Structured Input"],
        help="Choose between natural language or structured form input"
    )
    
    if query_mode == "Natural Language":
        st.subheader("Ask a Question")
        user_query = st.text_area(
            "Enter your question about weather, air quality, or solar output:",
            placeholder="e.g., What is the weather quality in Riyadh on 2024-01-15?",
            height=100
        )
        query_source = "natural"
    
    else:  # Structured Input
        st.subheader("Structured Query")
        
        cities = ['Riyadh', 'Jeddah', 'Dammam', 'Medina', 'Mecca', 'Khobar', 'Abha']
        selected_city = st.selectbox("Select City:", cities)
        
        query_date = st.date_input(
            "Select Date:",
            value=datetime.now().date(),
            min_value=datetime(2024, 1, 1).date(),
            max_value=datetime.now().date()
        )
        
        user_query = f"What is the weather quality in {selected_city} on {query_date.strftime('%Y-%m-%d')}?"
        query_source = "structured"
    
    # Execute Query
    execute_button = st.button("🚀 Execute Query", use_container_width=True, type="primary")

# ============================================================================
# MAIN CONTENT
# ============================================================================
if execute_button and user_query:
    with st.spinner("⏳ Processing query..."):
        result = rag.process_query(user_query)
    
    if result["status"] == "success":
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "📈 Details", "💭 Interpretations", "💬 LLM Response"])
        
        # ====== TAB 1: PREDICTIONS ======
        with tab1:
            st.subheader("🎯 Prediction Results")
            
            predictions = result["predictions"]
            
            if predictions:
                # Create metric columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    solar_kwh = predictions.get('solar_output_kwh', 0)
                    st.metric(
                        "☀️ Solar Output",
                        f"{solar_kwh:.2f} kWh",
                        delta=None,
                        help="Estimated daily solar energy output"
                    )
                    # Solar quality indicator
                    if solar_kwh < 80:
                        quality = "⚠️ Low"
                        color = "#d32f2f"
                    elif solar_kwh < 150:
                        quality = "✓ Medium"
                        color = "#ff9800"
                    else:
                        quality = "✓✓ High"
                        color = "#31a049"
                    st.markdown(f"<p style='color: {color}; font-weight: bold;'>Quality: {quality}</p>", unsafe_allow_html=True)
                
                with col2:
                    aqi_val = predictions.get('aqi_value', 0)
                    st.metric(
                        "💨 AQI Value",
                        f"{aqi_val:.0f}",
                        delta=None,
                        help="Air Quality Index (0-500)"
                    )
                    aqi_color = get_aqi_color(aqi_val)
                    # AQI category
                    if aqi_val < 50:
                        aqi_cat = "Good ✓"
                    elif aqi_val < 100:
                        aqi_cat = "Moderate ⚠️"
                    else:
                        aqi_cat = "Unhealthy ❌"
                    st.markdown(f"<p style='color: {aqi_color}; font-weight: bold;'>Category: {aqi_cat}</p>", unsafe_allow_html=True)
                
                with col3:
                    risk_level = predictions.get('aqi_risk_level', 'Unknown')
                    risk_color = get_risk_color(risk_level)
                    st.markdown(f"<p style='color: {risk_color}; font-weight: bold; font-size: 1.5em;'>🎯 Risk Level</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: {risk_color}; font-weight: bold; font-size: 1.3em;'>{risk_level}</p>", unsafe_allow_html=True)
                
                # Visualization
                st.subheader("Prediction Gauge")
                
                fig = go.Figure(data=[
                    go.Indicator(
                        mode="gauge+number",
                        value=aqi_val,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "AQI Level"},
                        gauge={
                            'axis': {'range': [None, 500]},
                            'bar': {'color': aqi_color},
                            'steps': [
                                {'range': [0, 50], 'color': "#e8f5e9"},
                                {'range': [50, 100], 'color': "#fff3e0"},
                                {'range': [100, 500], 'color': "#ffebee"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 150
                            }
                        }
                    )
                ])
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No predictions generated")
        
        # ====== TAB 2: DETAILS ======
        with tab2:
            st.subheader("📋 Retrieved Data Features")
            
            if result["data"]:
                # Create a formatted table
                data_dict = result["data"]
                
                # Group features
                weather_cols = ['temperature_2m_mean', 'relative_humidity_2m_mean', 'wind_speed_10m_mean', 
                               'cloud_cover_mean', 'precipitation_sum', 'shortwave_radiation_sum', 'sunshine_duration']
                air_cols = ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🌡️ Weather Features")
                    weather_data = {k: v for k, v in data_dict.items() if k in weather_cols}
                    for key, val in weather_data.items():
                        if isinstance(val, (int, float)):
                            st.metric(key.replace('_', ' ').title(), f"{val:.2f}")
                        else:
                            st.metric(key.replace('_', ' ').title(), val)
                
                with col2:
                    st.subheader("💨 Air Quality Features")
                    air_data = {k: v for k, v in data_dict.items() if k in air_cols}
                    for key, val in air_data.items():
                        if isinstance(val, (int, float)):
                            st.metric(key.replace('_', ' ').title(), f"{val:.2f}")
                        else:
                            st.metric(key.replace('_', ' ').title(), val)
        
        # ====== TAB 3: INTERPRETATIONS ======
        with tab3:
            st.subheader("📚 Knowledge Base Insights")
            
            if result["interpretations"]:
                for i, interp in enumerate(result["interpretations"], 1):
                    st.info(f"**Insight {i}:** {interp}")
            else:
                st.warning("No interpretations retrieved")
        
        # ====== TAB 4: LLM RESPONSE ======
        with tab4:
            st.subheader("💬 AI-Generated Summary")
            
            if result["llm_response"]:
                # Check if it's an error
                if "Error" in result["llm_response"]:
                    st.warning(f"⚠️ {result['llm_response']}")
                    st.info("""
                    **To enable AI responses:**
                    1. Open a new terminal
                    2. Run: `ollama serve`
                    3. In another terminal, run: `ollama pull llama2`
                    4. Then refresh this page
                    """)
                else:
                    st.success(result["llm_response"])
            else:
                st.info("Ollama LLM not available. See instructions above.")
    
    else:
        # Error handling
        st.error(f"❌ Query Failed: {result.get('error', 'Unknown error')}")
        st.info("""
        **Tips for better queries:**
        - Include a city name (Riyadh, Jeddah, Dammam, etc.)
        - Include a date (e.g., 2024-01-15 or today/tomorrow)
        - Example: "What is the weather in Riyadh on 2024-02-20?"
        """)

# ============================================================================
# EXAMPLE QUERIES SECTION
# ============================================================================
if not execute_button or not user_query:
    st.divider()
    st.subheader("📝 Example Queries")
    
    examples = [
        "What is the weather quality in Riyadh on 2024-01-15?",
        "Tell me about solar output and air quality in Jeddah today",
        "How are weather conditions in Dammam on 2024-02-20?"
    ]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏙️ Riyadh Weather", use_container_width=True):
            st.session_state.query = examples[0]
            st.rerun()
    
    with col2:
        if st.button("🏖️ Jeddah Today", use_container_width=True):
            st.session_state.query = examples[1]
            st.rerun()
    
    with col3:
        if st.button("🏜️ Dammam Forecast", use_container_width=True):
            st.session_state.query = examples[2]
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; margin-top: 3rem;'>
    <p>🌞 <strong>Solar Project RAG System v1.0</strong></p>
    <p>Powered by scikit-learn, XGBoost, Sentence Transformers & Llama 2</p>
    <p><em>For questions about weather, solar output, and air quality in Saudi Arabia</em></p>
</div>
""", unsafe_allow_html=True)
