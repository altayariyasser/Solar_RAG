"""Portable, lightweight RAG pipeline for the Solar Project Streamlit app."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
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
DEFAULT_DATASET = APP_DIR / "data" / "solar_dataset.csv"


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
    def dataset_candidates(self) -> List[Path]:
        """Return the plain dataset followed by its compressed fallback."""
        candidates = [self.dataset_path]
        if self.dataset_path.suffix != ".gz":
            candidates.append(
                self.dataset_path.with_name(f"{self.dataset_path.name}.gz")
            )
        return candidates

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
        failures = []

        for candidate in self.dataset_candidates:
            if not candidate.exists():
                failures.append(f"{candidate.name}: missing")
                continue
            if candidate.stat().st_size == 0:
                failures.append(f"{candidate.name}: empty")
                continue

            try:
                frame = pd.read_csv(candidate)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                failures.append(f"{candidate.name}: {exc}")
                continue

            if "Date" not in frame.columns:
                failures.append(f"{candidate.name}: missing Date column")
                continue

            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            frame = frame.dropna(subset=["Date"]).copy()
            if frame.empty:
                failures.append(f"{candidate.name}: no valid dates")
                continue

            self.dataset_path = candidate
            self.df_merged = frame
            return True

        details = "; ".join(failures)
        raise ValueError(
            "Unable to load the solar dataset. "
            f"Attempted: {details}. Commit a non-empty dataset to data/."
        )

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

        values = pd.DataFrame(
            [
                {
                    name: float(feature_dict.get(name, 0) or 0)
                    for name in self.feature_names
                }
            ],
            columns=self.feature_names,
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
            "Higher shortwave radiation and longer sunshine duration usually support more solar generation.",
            "Cloud cover and precipitation can reduce the solar energy available to photovoltaic panels.",
            "Strong wind can cool solar panels, but very high wind may require operational precautions.",
            "Photovoltaic modules commonly lose conversion efficiency as cell temperature rises.",
            "AQI below 50 indicates good air quality and minimal health risk.",
            "AQI from 50 to 100 indicates moderate air quality.",
            "AQI above 100 is unhealthy and outdoor exposure should be reduced.",
            "High cloud cover can significantly reduce solar radiation and panel output.",
            "Dust and airborne particles can reduce panel efficiency and worsen air quality.",
            "High temperatures may reduce photovoltaic conversion efficiency.",
            "A site assessment should consider radiation, cloud cover, heat, dust, shading, panel area, tilt, and system efficiency.",
            "The predicted daily energy is a model estimate for configurations represented in the training dataset, not a guaranteed system yield.",
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.embeddings = self.vectorizer.fit_transform(self.knowledge_items)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.embeddings)[0]
        indices = scores.argsort()[::-1][:top_k]
        return [self.knowledge_items[index] for index in indices]


class OllamaExplainer:
    """Generate grounded explanations through Ollama Cloud when configured."""

    DEFAULT_HOST = "https://ollama.com"
    DEFAULT_MODEL = "gpt-oss:20b"
    CONTEXT_FEATURES = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_mean",
        "cloud_cover_mean",
        "precipitation_sum",
        "shortwave_radiation_sum",
        "sunshine_duration",
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = (api_key or os.getenv("OLLAMA_API_KEY", "")).strip()
        self.host = (
            host or os.getenv("OLLAMA_HOST", self.DEFAULT_HOST)
        ).strip().rstrip("/")
        self.model = (model or os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)).strip()

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.host and self.model)

    def explain(
        self,
        user_query: str,
        city: str,
        date_str: str,
        data: Dict,
        predictions: Dict,
        interpretations: List[str],
        intents: List[str],
    ) -> Tuple[Optional[str], str, Optional[str]]:
        if not self.configured:
            return None, "not_configured", None

        observed = {
            key: data.get(key)
            for key in self.CONTEXT_FEATURES
            if data.get(key) is not None
        }
        prompt = (
            f"User question: {user_query}\n"
            f"Detected topics: {', '.join(intents)}\n"
            f"Location and date: {city}, {date_str}\n"
            f"Observed measurements: {json.dumps(observed, default=str)}\n"
            f"Model predictions: {json.dumps(predictions, default=str)}\n"
            "Retrieved domain guidance:\n- "
            + "\n- ".join(interpretations)
            + "\n\nAnswer the user's exact question first, then connect the relevant "
            "weather observations to the solar-energy and air-quality predictions. "
            "If the user asks whether the location is suitable for solar, discuss both "
            "supporting and limiting factors. Include the predicted daily solar energy "
            "when relevant. Use no more than 220 words, mention important uncertainty, "
            "and give one practical recommendation. Use only the supplied facts. This "
            "is historical analysis, so never describe it as a live forecast."
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a concise solar-energy and air-quality analyst. "
                        "Ground every statement in the supplied context."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 220},
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            explanation = str(body.get("message", {}).get("content", "")).strip()
            if explanation:
                return explanation, "ollama_cloud", None
        except urllib.error.HTTPError as exc:
            return (
                None,
                "unavailable",
                f"Ollama Cloud returned HTTP {exc.code}. Check the API key, model, and free-tier limits.",
            )
        except (
            json.JSONDecodeError,
            OSError,
            TimeoutError,
            urllib.error.URLError,
        ) as exc:
            return (
                None,
                "unavailable",
                f"Ollama Cloud request failed: {type(exc).__name__}.",
            )
        return None, "unavailable", "Ollama Cloud returned an empty explanation."


class SolarRAG:
    """Combine dataset retrieval, compact ML predictions, and local RAG context."""

    INTENT_KEYWORDS = {
        "weather": {
            "weather",
            "temperature",
            "hot",
            "cold",
            "humidity",
            "wind",
            "rain",
            "cloud",
            "sunshine",
        },
        "solar energy": {
            "solar",
            "energy",
            "electricity",
            "power",
            "output",
            "generation",
            "kwh",
            "panel",
            "photovoltaic",
            "pv",
        },
        "air quality": {
            "air",
            "aqi",
            "pollution",
            "pm10",
            "pm2.5",
            "dust",
            "healthy",
            "health",
        },
        "solar suitability": {
            "suitable",
            "suitability",
            "good location",
            "best",
            "recommend",
            "worth",
            "feasible",
        },
    }

    CITY_ALIASES = {
        "makkah": "Mecca",
        "madinah": "Medina",
    }

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        ollama_api_key: Optional[str] = None,
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ):
        self.data_loader = DataLoader(dataset_path)
        self.trainer = ModelTrainer(self.data_loader)
        self.kb = KnowledgeBase()
        self.explainer = OllamaExplainer(
            api_key=ollama_api_key,
            host=ollama_host,
            model=ollama_model,
        )
        self.ready = False

    def setup(self) -> bool:
        self.ready = self.data_loader.load_datasets()
        return self.ready

    def process_query(self, user_query: str) -> Dict:
        city, date_str = self._extract_location_date(user_query)
        intents = self._detect_intents(user_query)
        result = {
            "query": user_query,
            "status": "processing",
            "data": None,
            "predictions": {},
            "interpretations": [],
            "intents": intents,
            "city": city,
            "date": date_str,
            "llm_response": None,
            "llm_status": "not_configured",
            "llm_error": None,
        }

        if not city:
            result.update(
                status="error",
                error=f"Choose one of: {', '.join(self.data_loader.cities)}.",
            )
            return result

        if not date_str:
            result.update(
                status="error",
                error=(
                    "Include a date in your question, for example 2024-02-02. "
                    f"Available dates are {self.data_loader.min_date} through "
                    f"{self.data_loader.max_date}."
                ),
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
            f"{user_query} {' '.join(intents)} {city} "
            f"{predictions['solar_output_kwh']:.1f} {predictions['aqi_value']:.1f}"
        )
        interpretations = self.kb.retrieve(query_context, top_k=4)
        fallback_summary = self._build_fallback_summary(
            city,
            date_str,
            data,
            predictions,
            interpretations,
            intents,
        )
        llm_response, llm_status, llm_error = self.explainer.explain(
            user_query,
            city,
            date_str,
            data,
            predictions,
            interpretations,
            intents,
        )

        result.update(
            status="success",
            data=self._clean_values(data),
            predictions=predictions,
            interpretations=interpretations,
            llm_response=llm_response or fallback_summary,
            llm_status=llm_status,
            llm_error=llm_error,
        )
        return result

    @classmethod
    def _detect_intents(cls, query: str) -> List[str]:
        normalized = query.lower()
        intents = [
            intent
            for intent, keywords in cls.INTENT_KEYWORDS.items()
            if any(keyword in normalized for keyword in keywords)
        ]
        return intents or ["combined conditions"]

    @staticmethod
    def _number(data: Dict, key: str) -> Optional[float]:
        value = data.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_fallback_summary(
        self,
        city: str,
        date_str: str,
        data: Dict,
        predictions: Dict,
        interpretations: List[str],
        intents: List[str],
    ) -> str:
        temperature = self._number(data, "temperature_2m_mean")
        humidity = self._number(data, "relative_humidity_2m_mean")
        wind = self._number(data, "wind_speed_10m_mean")
        cloud = self._number(data, "cloud_cover_mean")
        rain = self._number(data, "precipitation_sum")

        weather_parts = []
        if temperature is not None:
            weather_parts.append(f"mean temperature {temperature:.1f} °C")
        if humidity is not None:
            weather_parts.append(f"humidity {humidity:.0f}%")
        if wind is not None:
            weather_parts.append(f"wind {wind:.1f} km/h")
        if cloud is not None:
            weather_parts.append(f"cloud cover {cloud:.0f}%")
        if rain is not None:
            weather_parts.append(f"precipitation {rain:.1f} mm")

        sentences = [f"For {city} on {date_str}:"]
        if "weather" in intents or "combined conditions" in intents:
            sentences.append("Weather observations were " + ", ".join(weather_parts) + ".")
        if any(
            intent in intents
            for intent in ("solar energy", "solar suitability", "combined conditions")
        ):
            sentences.append(
                f"The trained solar model estimates {predictions['solar_output_kwh']:.1f} "
                "kWh of daily output for configurations represented in the dataset."
            )
        if any(
            intent in intents
            for intent in ("air quality", "solar suitability", "combined conditions")
        ):
            sentences.append(
                f"The AQI model estimates {predictions['aqi_value']:.0f} "
                f"({predictions['aqi_risk_level']})."
            )
        sentences.append(interpretations[0])
        return " ".join(sentences)

    @staticmethod
    def _clean_values(data: Dict) -> Dict:
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, np.generic):
                value = value.item()
            if not (isinstance(value, float) and np.isnan(value)):
                cleaned[key] = value
        return cleaned

    def _extract_location_date(
        self, query: str
    ) -> Tuple[Optional[str], Optional[str]]:
        query_lower = query.lower()
        city = next(
            (name for name in self.data_loader.cities if name.lower() in query_lower),
            None,
        )
        if city is None:
            city = next(
                (
                    canonical
                    for alias, canonical in self.CITY_ALIASES.items()
                    if alias in query_lower
                ),
                None,
            )

        match = re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", query)
        if match:
            parsed_date = pd.to_datetime(
                match.group(0).replace("/", "-"),
                errors="coerce",
            )
            date_str = (
                parsed_date.strftime("%Y-%m-%d")
                if not pd.isna(parsed_date)
                else None
            )
        elif "tomorrow" in query_lower:
            date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "yesterday" in query_lower:
            date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        elif "today" in query_lower:
            date_str = datetime.now().strftime("%Y-%m-%d")
        else:
            date_str = None
        return city, date_str


def main():
    rag = SolarRAG()
    rag.setup()
    print(rag.process_query("Solar conditions in Riyadh on 2024-01-15"))


if __name__ == "__main__":
    main()
