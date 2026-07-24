# ☀️ Solar Project RAG System

A **Retrieval-Augmented Generation (RAG)** system for predicting solar energy output and air quality in Saudi Arabia. Combines weather data, air quality indices, and solar panel data with machine learning models and a knowledge base to provide intelligent predictions and interpretations.

## 🎯 Features

### Core Models
- **☀️ Solar Output Regression** - Predicts daily solar energy generation (kWh) with R² = 0.904
- **💨 AQI Value Regression** - Forecasts Air Quality Index with R² = 0.746
- **⚠️ AQI Risk Classification** - Classifies air quality as Good/Moderate/Unhealthy with 99.7% accuracy

### RAG System Components
- **Data Retrieval** - Queries weather, air quality, and solar datasets for specific locations and dates
- **Intelligent Prediction** - Uses trained ML models to forecast outcomes based on retrieved data
- **Knowledge Base** - 13+ domain-specific interpretations with semantic retrieval
- **Cosine Similarity Matching** - Retrieves relevant insights using sentence embeddings
- **Llama 2 Integration** - Optional AI-generated summaries via Ollama

### Web Interface
- 🌐 **Interactive Streamlit UI** with dark theme
- 📊 **Real-time visualizations** (gauges, metrics, charts)
- 🔍 **Natural language or structured queries**
- 📋 **Detailed data inspection** and prediction breakdown
- 💭 **Knowledge base insights** with semantic retrieval
- 💬 **AI-generated summaries** (when Ollama is running)

## 📊 Data

The system uses three integrated datasets:
- **Weather Data**: 1,830 records of daily weather (temperature, humidity, cloud cover, radiation, etc.)
- **Air Quality Data**: 1,830 records of pollutants (PM10, PM2.5, NO2, O3, CO, SO2, AQI)
- **Solar Data**: 10,980 records of panel configurations and output measurements

**Coverage**: Saudi Arabia (Riyadh, Jeddah, Dammam, Medina, Mecca, Khobar, Abha) | Date Range: 2024

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- (Optional) Ollama for LLM features

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/solar-rag-system.git
cd solar-rag-system
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### Option A: Command-Line Interface
```bash
python rag.py
```

Example queries automatically processed:
- "What is the weather quality in Riyadh on 2024-01-15?"
- "Tell me about air quality and solar output in Jeddah today"
- "How are weather conditions in Dammam on 2024-02-20?"

#### Option B: Interactive Web Interface
```bash
streamlit run rag_app.py
```

Open browser → http://localhost:8501

Choose between:
- **Natural Language Mode**: Ask questions in plain English
- **Structured Mode**: Select city and date from dropdowns

### Option C: Enable Ollama (AI Summaries)

1. **Install Ollama** from https://ollama.ai
2. **Start Ollama server** in a separate terminal:
```bash
ollama serve
```
3. **Pull Llama2 model**:
```bash
ollama pull llama2
```
4. **Refresh the web interface** - AI responses will now be generated

## 📁 Project Structure

```
solar-rag-system/
├── rag.py                 # Core RAG system & models
├── rag_app.py            # Streamlit web interface
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── LICENSE              # MIT License
└── Final_Dataset*.csv   # Merged dataset (loaded from parent folder)
```

## 🏗️ Architecture

### Data Pipeline
```
Weather Data + Air Quality Data + Solar Data
           ↓
    Merge & Clean
           ↓
    Feature Engineering (14 features)
           ↓
    Train-Test Split (80-20)
           ↓
    Scale Features (StandardScaler)
```

### Model Pipeline
```
Input Query → Extract Location/Date → Retrieve Data → Feature Extraction
                                           ↓
                    ┌────────────────────────┼────────────────────────┐
                    ↓                        ↓                        ↓
            Solar Regression        AQI Regression          AQI Classification
            (RandomForest)          (LinearRegression)      (RandomForest)
                    ↓                        ↓                        ↓
                 Prediction 1         Prediction 2           Prediction 3
                    ↓                        ↓                        ↓
                    └────────────────────────┼────────────────────────┘
                                    ↓
                        Retrieve from Knowledge Base
                              (Embeddings)
                                    ↓
                        Generate Interpretation
                                    ↓
                        Optional: LLM Response
```

### Knowledge Base
- **Solar Interpretations**: Relates output levels to weather conditions
- **AQI Interpretations**: Maps numeric values to health risk levels
- **Risk Classifications**: Explains Good/Moderate/Unhealthy categories
- **Weather Relationships**: Links weather features to pollutant formation

## 🎓 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Solar Output Regression | R² Score | 0.904 |
| Solar Output Regression | MSE | 0.53 |
| AQI Value Regression | R² Score | 0.746 |
| AQI Value Regression | MSE | 10,623.69 |
| AQI Risk Classification | Accuracy | 99.7% |

## 🔬 Technical Stack

**Machine Learning**
- scikit-learn (regression, classification, preprocessing)
- XGBoost (initial classification experiments)
- RandomForest (primary models)

**NLP & Embeddings**
- Sentence Transformers (all-MiniLM-L6-v2)
- Cosine Similarity (PyTorch)

**Data Processing**
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly

**Web Framework**
- Streamlit (UI)
- Ollama (LLM integration via API)

**Deployment Ready**
- Docker support (optional)
- Streamlit Cloud compatible
- GitHub Actions CI/CD (optional)

## 📝 Example Queries

### Natural Language Examples
```
"What is the weather quality in Riyadh on 2024-01-15?"
"How is the air quality in Jeddah today?"
"Tell me about solar output potential in Dammam on 2024-02-20"
"What are weather conditions for Medina on tomorrow?"
```

### Expected Response Structure
```
{
  "query": "...",
  "status": "success",
  "data": { weather and air quality features },
  "predictions": {
    "solar_output_kwh": 120.45,
    "aqi_value": 75.32,
    "aqi_risk_level": "Moderate"
  },
  "interpretations": [
    "Insight 1 from knowledge base",
    "Insight 2 from knowledge base",
    "Insight 3 from knowledge base"
  ],
  "llm_response": "Natural language summary..." (if Ollama available)
}
```

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended for Web UI)
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with one click
4. Note: Data files must be accessible (use cloud storage or commit non-sensitive data)

### Option 2: Docker Deployment
```bash
# Build image
docker build -t solar-rag .

# Run container
docker run -p 8501:8501 solar-rag
```

### Option 3: Local Server
```bash
# Start Streamlit server
streamlit run rag_app.py --server.port 8080 --server.address 0.0.0.0
```

### Option 4: Traditional VPS/EC2
Clone repo → Install dependencies → Run with supervisor/systemd

## ⚙️ Configuration

### Environment Variables
Create `.streamlit/config.toml` for Streamlit settings:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
font = "sans serif"

[server]
headless = true
port = 8501
```

### Model Parameters
Edit hyperparameters in `rag.py`:
- `RandomForest` trees: `n_estimators=100`
- `RandomForest` depth: `max_depth=10`
- Training/test split: `test_size=0.2`

## 🐛 Troubleshooting

### "No data found for [city] on [date]"
- Ensure date is within 2024 (current dataset range)
- Check city name spelling (case-insensitive)

### "Ollama not available"
- Install Ollama from https://ollama.ai
- Run `ollama serve` in separate terminal
- Run `ollama pull llama2`

### Module import errors
```bash
pip install --upgrade -r requirements.txt
```

### Streamlit performance issues
```bash
# Clear cache
streamlit cache clear

# Run with reduced logging
streamlit run rag_app.py --logger.level=error
```

## 📚 Dataset Information

**Weather Features (8)**
- temperature_2m_mean
- relative_humidity_2m_mean
- surface_pressure_mean
- wind_speed_10m_mean
- cloud_cover_mean
- precipitation_sum
- shortwave_radiation_sum
- sunshine_duration

**Air Quality Features (6)**
- pm10
- pm2_5
- carbon_monoxide
- nitrogen_dioxide
- ozone
- sulphur_dioxide

**Targets**
- Estimated Daily Output (kWh) - Solar regression
- us_aqi - AQI regression & classification

## 📄 License

MIT License - See LICENSE file for details

## 👥 Contributors

- Team: Atheer, Ritaj, Yasser, Hussam

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📧 Support

For issues, questions, or suggestions:
- Open an Issue on GitHub
- Contact the development team

## 🔮 Future Enhancements

- [ ] Multi-day forecasting
- [ ] Historical trend analysis
- [ ] Mobile app integration
- [ ] Real-time API data ingestion
- [ ] Advanced time-series models (LSTM, Prophet)
- [ ] User preferences & saved queries
- [ ] Batch prediction export
- [ ] Integration with renewable energy platforms

## 📖 References

- Sentence Transformers: https://www.sbert.net/
- Ollama: https://ollama.ai/
- Streamlit: https://streamlit.io/
- scikit-learn: https://scikit-learn.org/

---

**Last Updated**: 2026-07-24  
**Status**: ✅ Production Ready
