"""Portable, lightweight RAG pipeline for the Solar Project Streamlit app."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


APP_DIR = Path(__file__).resolve().parent

# The repo has historically had the real data under a couple of different
# names/locations ("data/solar_dataset.csv" was left as an empty placeholder
# in some commits). Try the known candidates in order and use the first one
# that actually exists and has content, instead of hard-failing on a single
# empty file.
_CANDIDATE_DATASETS = [
    APP_DIR / "data" / "Final_Dataset.csv",
    APP_DIR / "Final_Dataset.csv",
    APP_DIR / "data" / "solar_dataset.csv",
]


def _resolve_default_dataset() -> Path:
    for candidate in _CANDIDATE_DATASETS:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    # Fall back to the first candidate so the error message below still
    # points somewhere sensible if none of them exist.
    return _CANDIDATE_DATASETS[0]


DEFAULT_DATASET = _resolve_default_dataset()


class DataLoader:
    """Load the repository dataset using paths that work on Windows and Linux."""

    CITY_COLUMNS = {
        "Jeddah": "City_Jeddah",
        "Mecca": "City_Mecca",
        "Medina": "City_Medina",
        "Riyadh": "City_Riyadh",
    }
    BASE_CITY = "Dammam"

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = Path(dataset_path or DEFAULT_DATASET)
        self.df_merged: Optional[pd.DataFrame] = None

    @property
    def cities(self) -> List[str]:
        return [*self.CITY_COLUMNS, self.BASE_CITY]

    @property
    def min_date(self):
        return self.df_merged["Date"].min().date() if self.df_merged is not None else None

    @property
    def max_date(self):
        return self.df_merged["Date"].max().date() if self.df_merged is not None else None

    def load_datasets(self) -> bool:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. "
                "Commit data/solar_dataset.csv to the GitHub repository."
            )

        frame = pd.read_csv(self.dataset_path)
        if "Date" not in frame.columns:
            raise ValueError("The dataset must contain a Date column.")

        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).copy()
        if frame.empty:
            raise ValueError("The dataset contains no valid dates.")

        self.df_merged = frame
        return True

    def _city_mask(self, city: str) -> pd.Series:
        assert self.df_merged is not None
        canonical = next((name for name in self.cities if name.lower() == city.lower()), None)
        if canonical is None:
            return pd.Series(False, index=self.df_merged.index)

        if canonical in self.CITY_COLUMNS:
            column = self.CITY_COLUMNS[canonical]
            return self.df_merged[column].fillna(0).astype(int).eq(1)

        encoded_columns = [
            column for column in self.CITY_COLUMNS.values() if column in self.df_merged.columns
        ]
        return self.df_merged[encoded_columns].fillna(0).sum(axis=1).eq(0)

    def get_data_for_location_date(self, city: str, date_str: str) -> Optional[Dict]:
        if self.df_merged is None:
            return None

        target_date = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(target_date):
            return None

        matches = self.df_merged[
            self._city_mask(city) & self.df_merged["Date"].dt.date.eq(target_date.date())
        ]
        if matches.empty:
            return None

        # Each city/date has multiple panel configurations. Average their numeric
        # values so the query represents the city-day rather than one panel row.
        values = matches.select_dtypes(include=[np.number]).mean().to_dict()
        values["Date"] = target_date.strftime("%Y-%m-%d")
        values["City"] = city.title()
        return values


class ModelTrainer:
    """Train compact local models only when the first query is submitted."""

    WEATHER_FEATURES = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "surface_pressure_mean",
        "wind_speed_10m_mean",
        "cloud_cover_mean",
        "precipitation_sum",
        "shortwave_radiation_sum",
        "sunshine_duration",
    ]
    AIR_FEATURES = [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
    ]

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.model_solar_reg: Optional[RandomForestRegressor] = None
        self.model_aqi_reg: Optional[LinearRegression] = None
        self.model_aqi_class: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._trained = False

    def train_models(self) -> bool:
        if self._trained:
            return True
        if self.data_loader.df_merged is None:
            return False

        frame = self.data_loader.df_merged
        self.feature_names = [
            name
            for name in self.WEATHER_FEATURES + self.AIR_FEATURES
            if name in frame.columns
        ]
        required_targets = {"Estimated Daily Output (kWh)", "us_aqi"}
        if not self.feature_names or not required_targets.issubset(frame.columns):
            return False

        training = frame[self.feature_names + list(required_targets)].copy()
        training = training.apply(pd.to_numeric, errors="coerce")
        training = training.fillna(training.median(numeric_only=True)).dropna()
        if len(training) > 6000:
            training = training.sample(6000, random_state=42)

        x_scaled = self.scaler.fit_transform(training[self.feature_names])
        y_solar = training["Estimated Daily Output (kWh)"].to_numpy()
        y_aqi = training["us_aqi"].to_numpy()
        y_class = pd.cut(
            y_aqi,
            bins=[-np.inf, 50, 100, np.inf],
            labels=[0, 1, 2],
        ).astype(int)

        self.model_solar_reg = RandomForestRegressor(
            n_estimators=40,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ).fit(x_scaled, y_solar)
        self.model_aqi_reg = LinearRegression().fit(x_scaled, y_aqi)
        self.model_aqi_class = RandomForestClassifier(
            n_estimators=40,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ).fit(x_scaled, y_class)
        self._trained = True
        return True

    def predict(self, feature_dict: Dict) -> Dict:
        if not self.train_models():
            return {}

        values = np.array(
            [[float(feature_dict.get(name, 0) or 0) for name in self.feature_names]]
        )
        scaled = self.scaler.transform(values)
        risk_names = {0: "Good", 1: "Moderate", 2: "Unhealthy"}

        return {
            "solar_output_kwh": float(max(0, self.model_solar_reg.predict(scaled)[0])),
            "aqi_value": float(max(0, self.model_aqi_reg.predict(scaled)[0])),
            "aqi_risk_level": risk_names.get(
                int(self.model_aqi_class.predict(scaled)[0]), "Unknown"
            ),
        }


class KnowledgeBase:
    """Small local knowledge base retrieved with TF-IDF cosine similarity."""

    def __init__(self):
        self.knowledge_items = [
            "Solar output above 150 kWh indicates strong generation conditions.",
            "Solar output between 80 and 150 kWh indicates moderate generation conditions.",
            "Solar output below 80 kWh can result from cloud, dust, or weak radiation.",
            "AQI below 50 indicates good air quality and minimal health risk.",
            "AQI from 50 to 100 indicates moderate air quality.",
            "AQI above 100 is unhealthy and outdoor exposure should be reduced.",
            "High cloud cover can significantly reduce solar radiation and panel output.",
            "Dust and airborne particles can reduce panel efficiency and worsen air quality.",
            "High temperatures may reduce photovoltaic conversion efficiency.",
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.embeddings = self.vectorizer.fit_transform(self.knowledge_items)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.embeddings)[0]
        indices = scores.argsort()[::-1][:top_k]
        return [self.knowledge_items[index] for index in indices]


class SolarRAG:
    """Combine dataset retrieval, compact ML predictions, and local RAG context."""

    def __init__(self, dataset_path: Optional[Path] = None):
        self.data_loader = DataLoader(dataset_path)
        self.trainer = ModelTrainer(self.data_loader)
        self.kb = KnowledgeBase()
        self.ready = False

    def setup(self) -> bool:
        self.ready = self.data_loader.load_datasets()
        return self.ready

    def process_query(self, user_query: str) -> Dict:
        city, date_str = self._extract_location_date(user_query)
        result = {
            "query": user_query,
            "status": "processing",
            "data": None,
            "predictions": {},
            "interpretations": [],
            "llm_response": None,
        }

        if not city:
            result.update(
                status="error",
                error=f"Choose one of: {', '.join(self.data_loader.cities)}.",
            )
            return result

        data = self.data_loader.get_data_for_location_date(city, date_str)
        if not data:
            result.update(
                status="error",
                error=(
                    f"No data found for {city} on {date_str}. "
                    f"Available dates are {self.data_loader.min_date} through "
                    f"{self.data_loader.max_date}."
                ),
            )
            return result

        predictions = self.trainer.predict(data)
        if not predictions:
            result.update(status="error", error="The prediction models could not be trained.")
            return result

        query_context = (
            f"solar weather air quality AQI {city} "
            f"{predictions['solar_output_kwh']:.1f} {predictions['aqi_value']:.1f}"
        )
        interpretations = self.kb.retrieve(query_context)
        summary = (
            f"For {city} on {date_str}, estimated solar output is "
            f"{predictions['solar_output_kwh']:.1f} kWh and predicted AQI is "
            f"{predictions['aqi_value']:.0f} ({predictions['aqi_risk_level']}). "
            f"{interpretations[0]}"
        )

        result.update(
            status="success",
            data=self._clean_values(data),
            predictions=predictions,
            interpretations=interpretations,
            llm_response=summary,
        )
        return result

    @staticmethod
    def _clean_values(data: Dict) -> Dict:
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, np.generic):
                value = value.item()
            if not (isinstance(value, float) and np.isnan(value)):
                cleaned[key] = value
        return cleaned

    def _extract_location_date(self, query: str) -> Tuple[Optional[str], str]:
        query_lower = query.lower()
        city = next(
            (name for name in self.data_loader.cities if name.lower() in query_lower),
            None,
        )

        match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", query)
        if match:
            date_str = match.group(0)
        elif "tomorrow" in query_lower:
            date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "yesterday" in query_lower:
            date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")
        return city, date_str


def main():
    rag = SolarRAG()
    rag.setup()
    print(rag.process_query("Solar conditions in Riyadh on 2024-01-15"))


if __name__ == "__main__":
    main()
