"""
Solar Project RAG System
Retrieves weather data for specific locations/dates, makes predictions using trained models,
and interprets results using a knowledge base with cosine similarity retrieval.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML and Embedding Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sentence_transformers import SentenceTransformer, util
import torch

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama not installed. Using only local inference.")


# ============================================================================
# 1. DATA LOADER
# ============================================================================
class DataLoader:
    """Load and merge datasets from archive and current folders"""
    
    def __init__(self):
        self.archive_path = r"c:\Users\altay\Downloads\archive (14)"
        self.current_path = r"c:\Users\altay\Downloads\New folder"
        self.df_weather = None
        self.df_air = None
        self.df_solar = None
        self.df_merged = None
    
    def load_datasets(self) -> bool:
        """Load all three datasets"""
        try:
            print("📂 Loading datasets...")
            
            # Load from archive
            weather_path = os.path.join(self.archive_path, "Weather_Dataset .csv")
            air_path = os.path.join(self.archive_path, "AirQuality_Dataset .csv")
            solar_path = os.path.join(self.archive_path, "ksa_solar_dataset_2024_detailed.csv")
            
            if os.path.exists(weather_path):
                self.df_weather = pd.read_csv(weather_path)
                print(f"✓ Weather data: {self.df_weather.shape}")
            
            if os.path.exists(air_path):
                self.df_air = pd.read_csv(air_path)
                print(f"✓ Air Quality data: {self.df_air.shape}")
            
            if os.path.exists(solar_path):
                self.df_solar = pd.read_csv(solar_path, encoding="latin1")
                print(f"✓ Solar data: {self.df_solar.shape}")
            
            # Load merged dataset if available
            final_path = os.path.join(self.current_path, "Final_Dataset (1).csv")
            if os.path.exists(final_path):
                self.df_merged = pd.read_csv(final_path)
                print(f"✓ Merged dataset: {self.df_merged.shape}")
                return True
            
            return bool(self.df_weather is not None and self.df_air is not None)
        
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return False
    
    def get_data_for_location_date(self, city: str, date_str: str) -> Optional[Dict]:
        """Retrieve data for a specific location and date"""
        try:
            target_date = pd.to_datetime(date_str)
            
            if self.df_merged is not None:
                # Use merged dataset if available
                if 'City' in self.df_merged.columns and 'Date' in self.df_merged.columns:
                    self.df_merged['Date'] = pd.to_datetime(self.df_merged['Date'], errors='coerce')
                    record = self.df_merged[
                        (self.df_merged['City'].str.lower() == city.lower()) &
                        (self.df_merged['Date'].dt.date == target_date.date())
                    ]
                    if not record.empty:
                        return record.iloc[0].to_dict()
            
            # Fallback: query individual datasets
            data = {}
            if self.df_weather is not None and 'Date' in self.df_weather.columns:
                self.df_weather['Date'] = pd.to_datetime(self.df_weather['Date'], errors='coerce')
                w_record = self.df_weather[
                    (self.df_weather['City'].str.lower() == city.lower()) &
                    (self.df_weather['Date'].dt.date == target_date.date())
                ]
                if not w_record.empty:
                    data.update(w_record.iloc[0].to_dict())
            
            if self.df_air is not None and 'Date' in self.df_air.columns:
                self.df_air['Date'] = pd.to_datetime(self.df_air['Date'], errors='coerce')
                a_record = self.df_air[
                    (self.df_air['City'].str.lower() == city.lower()) &
                    (self.df_air['Date'].dt.date == target_date.date())
                ]
                if not a_record.empty:
                    data.update(a_record.iloc[0].to_dict())
            
            return data if data else None
        
        except Exception as e:
            print(f"❌ Error retrieving data: {e}")
            return None


# ============================================================================
# 2. MODEL TRAINER
# ============================================================================
class ModelTrainer:
    """Train the three models: Solar Regression, AQI Regression, AQI Classification"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.model_solar_reg = None
        self.model_aqi_reg = None
        self.model_aqi_class = None
        self.scaler = StandardScaler()
        self.feature_names = {}
    
    def prepare_features(self) -> Tuple[np.ndarray, Dict]:
        """Extract and prepare features from merged dataset"""
        try:
            df = self.data_loader.df_merged
            
            # Define features for each model
            weather_features = [
                'temperature_2m_mean', 'relative_humidity_2m_mean',
                'surface_pressure_mean', 'wind_speed_10m_mean',
                'cloud_cover_mean', 'precipitation_sum',
                'shortwave_radiation_sum', 'sunshine_duration'
            ]
            
            air_features = [
                'pm10', 'pm2_5', 'carbon_monoxide',
                'nitrogen_dioxide', 'ozone', 'sulphur_dioxide'
            ]
            
            # Combine features
            available_features = [f for f in weather_features + air_features if f in df.columns]
            
            self.feature_names['solar'] = available_features
            self.feature_names['aqi_reg'] = available_features
            self.feature_names['aqi_class'] = available_features
            
            X = df[available_features].fillna(df[available_features].mean())
            return X.values, {'features': available_features, 'shape': X.shape}
        
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return None, None
    
    def train_models(self) -> bool:
        """Train all three models"""
        try:
            print("\n🔧 Training models...")
            df = self.data_loader.df_merged
            
            X, info = self.prepare_features()
            if X is None:
                return False
            
            print(f"📊 Features prepared: {info}")
            
            # Targets
            y_solar = None
            y_aqi_value = None
            y_aqi_class = None
            
            if 'Estimated Daily Output (kWh)' in df.columns:
                y_solar = df['Estimated Daily Output (kWh)'].fillna(df['Estimated Daily Output (kWh)'].mean()).values
            
            if 'us_aqi' in df.columns:
                y_aqi_value = df['us_aqi'].fillna(df['us_aqi'].mean()).values
                # Create AQI risk classes: 0=Good, 1=Moderate, 2=Unhealthy
                aqi_filled = pd.Series(y_aqi_value).fillna(pd.Series(y_aqi_value).mean())
                y_aqi_class = pd.cut(aqi_filled, bins=[0, 50, 100, 500], labels=[0, 1, 2], include_lowest=True).fillna(2).astype(int).values
            
            # Split data
            if y_solar is not None and y_aqi_value is not None and y_aqi_class is not None:
                X_train, X_test, y_train_solar, y_test_solar, y_train_aqi, y_test_aqi, y_train_class, y_test_class = \
                    train_test_split(X, y_solar, y_aqi_value, y_aqi_class, test_size=0.2, random_state=42)
            else:
                print("❌ Missing targets for training")
                return False
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Model 1: Solar Regression (Linear + RandomForest ensemble)
            self.model_solar_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            self.model_solar_reg.fit(X_train_scaled, y_train_solar)
            pred = self.model_solar_reg.predict(X_test_scaled)
            mse = mean_squared_error(y_test_solar, pred)
            r2 = r2_score(y_test_solar, pred)
            print(f"  ✓ Solar Output Regression - R²: {r2:.3f}, MSE: {mse:.2f}")
            
            # Model 2: AQI Value Regression
            self.model_aqi_reg = LinearRegression()
            self.model_aqi_reg.fit(X_train_scaled, y_train_aqi)
            pred = self.model_aqi_reg.predict(X_test_scaled)
            mse = mean_squared_error(y_test_aqi, pred)
            r2 = r2_score(y_test_aqi, pred)
            print(f"  ✓ AQI Value Regression - R²: {r2:.3f}, MSE: {mse:.2f}")
            
            # Model 3: AQI Risk Classification
            self.model_aqi_class = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            self.model_aqi_class.fit(X_train_scaled, y_train_class)
            pred = self.model_aqi_class.predict(X_test_scaled)
            acc = accuracy_score(y_test_class, pred)
            print(f"  ✓ AQI Risk Classification - Accuracy: {acc:.3f}")
            
            return True
        
        except Exception as e:
            print(f"❌ Error training models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, feature_dict: Dict) -> Dict:
        """Make predictions using all three models"""
        try:
            # Extract features in correct order
            feature_values = []
            for feat in self.feature_names.get('solar', []):
                feature_values.append(feature_dict.get(feat, 0))
            
            X = np.array([feature_values])
            X_scaled = self.scaler.transform(X)
            
            predictions = {}
            
            if self.model_solar_reg is not None:
                pred_solar = self.model_solar_reg.predict(X_scaled)[0]
                predictions['solar_output_kwh'] = float(max(0, pred_solar))
            
            if self.model_aqi_reg is not None:
                pred_aqi_val = self.model_aqi_reg.predict(X_scaled)[0]
                predictions['aqi_value'] = float(max(0, pred_aqi_val))
            
            if self.model_aqi_class is not None:
                pred_aqi_class = self.model_aqi_class.predict(X_scaled)[0]
                class_names = {0: 'Good', 1: 'Moderate', 2: 'Unhealthy'}
                predictions['aqi_risk_level'] = class_names.get(int(pred_aqi_class), 'Unknown')
            
            return predictions
        
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return {}


# ============================================================================
# 3. KNOWLEDGE BASE & RETRIEVAL
# ============================================================================
class KnowledgeBase:
    """Knowledge base for interpreting predictions using embeddings and cosine similarity"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_items = []
        self.embeddings = None
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize domain-specific knowledge about interpretations"""
        self.knowledge_items = [
            # Solar Output Interpretations
            {
                "type": "solar",
                "text": "High solar output above 150 kWh indicates excellent weather conditions with low cloud cover and strong radiation.",
                "range": (150, float('inf'))
            },
            {
                "type": "solar",
                "text": "Medium solar output between 80-150 kWh shows good weather conditions suitable for solar generation.",
                "range": (80, 150)
            },
            {
                "type": "solar",
                "text": "Low solar output below 80 kWh indicates cloudy weather or dust storms reducing solar radiation.",
                "range": (0, 80)
            },
            
            # AQI Value Interpretations
            {
                "type": "aqi_value",
                "text": "AQI below 50 indicates good air quality with minimal pollutants and low health risks.",
                "range": (0, 50)
            },
            {
                "type": "aqi_value",
                "text": "AQI 50-100 indicates moderate air quality where sensitive groups may experience health effects.",
                "range": (50, 100)
            },
            {
                "type": "aqi_value",
                "text": "AQI 100-150 indicates unhealthy air quality. General public may experience respiratory effects.",
                "range": (100, 150)
            },
            {
                "type": "aqi_value",
                "text": "AQI above 150 indicates very unhealthy air quality. All groups at risk, outdoor activities not recommended.",
                "range": (150, float('inf'))
            },
            
            # Risk Level Interpretations
            {
                "type": "aqi_risk",
                "text": "Good air quality means the weather conditions are favorable for outdoor activities and solar operations.",
                "level": "Good"
            },
            {
                "type": "aqi_risk",
                "text": "Moderate air quality means vulnerable populations should limit outdoor exposure, but general operations can continue.",
                "level": "Moderate"
            },
            {
                "type": "aqi_risk",
                "text": "Unhealthy air quality means air pollution is a serious concern affecting public health and solar panel efficiency.",
                "level": "Unhealthy"
            },
            
            # Weather-Related Knowledge
            {
                "type": "weather",
                "text": "High cloud cover (>70%) significantly reduces solar radiation and therefore solar panel output.",
                "topic": "cloud_cover"
            },
            {
                "type": "weather",
                "text": "Low precipitation helps maintain solar panel efficiency by reducing dust accumulation.",
                "topic": "precipitation"
            },
            {
                "type": "weather",
                "text": "High humidity combined with temperature fluctuations can increase air pollutants in Saudi Arabia.",
                "topic": "humidity"
            }
        ]
        
        # Embed all knowledge items
        texts = [item['text'] for item in self.knowledge_items]
        self.embeddings = self.model.encode(texts, convert_to_tensor=True)
        print(f"✓ Knowledge base initialized with {len(self.knowledge_items)} items")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k most relevant knowledge items using cosine similarity"""
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(similarities, k=min(top_k, len(self.knowledge_items)))
        
        results = []
        for idx in top_results.indices:
            results.append(self.knowledge_items[int(idx)]['text'])
        
        return results


# ============================================================================
# 4. RAG PIPELINE
# ============================================================================
class SolarRAG:
    """Main RAG system combining retrieval, prediction, and interpretation"""
    
    def __init__(self):
        print("🚀 Initializing Solar Project RAG System...\n")
        
        self.data_loader = DataLoader()
        self.trainer = ModelTrainer(self.data_loader)
        self.kb = KnowledgeBase()
        self.llm_available = OLLAMA_AVAILABLE
    
    def setup(self) -> bool:
        """Initialize and train the system"""
        if not self.data_loader.load_datasets():
            print("❌ Failed to load datasets")
            return False
        
        if not self.trainer.train_models():
            print("❌ Failed to train models")
            return False
        
        print("\n✅ RAG System Ready!\n")
        return True
    
    def process_query(self, user_query: str) -> Dict:
        """End-to-end query processing: extract -> retrieve -> predict -> interpret"""
        print(f"\n{'='*70}")
        print(f"🔍 Query: {user_query}")
        print(f"{'='*70}\n")
        
        result = {
            "query": user_query,
            "status": "processing",
            "data": None,
            "predictions": None,
            "interpretations": [],
            "llm_response": None
        }
        
        try:
            # Step 1: Extract location and date from query
            city, date_str = self._extract_location_date(user_query)
            print(f"📍 Extracted: City={city}, Date={date_str}")
            
            if not city or not date_str:
                result["status"] = "error"
                result["error"] = "Could not extract city and date from query"
                return result
            
            # Step 2: Retrieve data
            data = self.data_loader.get_data_for_location_date(city, date_str)
            if not data:
                result["status"] = "error"
                result["error"] = f"No data found for {city} on {date_str}"
                return result
            
            result["data"] = {k: v for k, v in data.items() if isinstance(v, (int, float, str))}
            print(f"✓ Data retrieved: {len(data)} features")
            
            # Step 3: Make predictions
            predictions = self.trainer.predict(data)
            result["predictions"] = predictions
            print(f"✓ Predictions made:")
            for key, val in predictions.items():
                print(f"  - {key}: {val}")
            
            # Step 4: Retrieve interpretations from knowledge base
            interpretation_query = f"solar output weather conditions air quality AQI {city}"
            interpretations = self.kb.retrieve(interpretation_query, top_k=3)
            result["interpretations"] = interpretations
            print(f"\n📚 Retrieved {len(interpretations)} interpretations:")
            for i, interp in enumerate(interpretations, 1):
                print(f"  {i}. {interp}")
            
            # Step 5: Generate LLM response if available
            if self.llm_available:
                result["llm_response"] = self._generate_llm_response(
                    user_query, data, predictions, interpretations
                )
                print(f"\n💬 LLM Response:\n{result['llm_response']}")
            else:
                print("\n⚠️  Ollama not available. Install with: ollama serve &")
                print("Then run: ollama pull llama2")
            
            result["status"] = "success"
        
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def _extract_location_date(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract city name and date from natural language query"""
        query_lower = query.lower()
        
        # List of known cities in KSA
        cities = ['riyadh', 'jeddah', 'dammam', 'medina', 'mecca', 'khobar', 'abha']
        city = None
        for c in cities:
            if c in query_lower:
                city = c.capitalize()
                break
        
        # Try to extract date
        date_str = None
        from dateutil import parser
        
        # Look for date patterns
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(0)
                    break
                except:
                    pass
        
        # If no specific date, use today
        if not date_str:
            if 'today' in query_lower or 'now' in query_lower:
                date_str = datetime.now().strftime('%Y-%m-%d')
            elif 'tomorrow' in query_lower:
                date_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            elif 'yesterday' in query_lower:
                date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')
        
        return city, date_str
    
    def _generate_llm_response(self, query: str, data: Dict, predictions: Dict, interpretations: List[str]) -> str:
        """Generate natural language response using Llama via Ollama"""
        try:
            # Prepare context
            context = f"""
User Question: {query}

Available Data:
- Temperature: {data.get('temperature_2m_mean', 'N/A')}°C
- Humidity: {data.get('relative_humidity_2m_mean', 'N/A')}%
- Cloud Cover: {data.get('cloud_cover_mean', 'N/A')}%
- Wind Speed: {data.get('wind_speed_10m_mean', 'N/A')} m/s
- PM2.5: {data.get('pm2_5', 'N/A')} µg/m³
- PM10: {data.get('pm10', 'N/A')} µg/m³

Predictions:
- Estimated Solar Output: {predictions.get('solar_output_kwh', 'N/A')} kWh
- Predicted AQI: {predictions.get('aqi_value', 'N/A')}
- Air Quality Risk Level: {predictions.get('aqi_risk_level', 'N/A')}

Knowledge Base Insights:
{chr(10).join([f'- {i}' for i in interpretations])}

Please provide a brief, actionable response about weather conditions, solar generation potential, and air quality.
"""
            
            response = ollama.generate(
                model='llama2',
                prompt=context,
                stream=False
            )
            
            return response.get('response', 'No response generated')
        
        except Exception as e:
            return f"LLM Generation Error: {str(e)}"


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
def main():
    """Main entry point"""
    rag = SolarRAG()
    
    if not rag.setup():
        print("❌ Setup failed")
        return
    
    # Example queries
    queries = [
        "What is the weather quality in Riyadh on 2024-01-15?",
        "Tell me about air quality and solar output in Jeddah today",
        "How are weather conditions in Dammam on 2024-02-20?"
    ]
    
    for query in queries:
        result = rag.process_query(query)
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
