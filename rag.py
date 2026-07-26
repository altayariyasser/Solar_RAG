"""Portable RAG pipeline used by the integrated SolarIQ Streamlit app."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from functools import lru_cache
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

CITY_COORDINATES = {
    "Riyadh": (24.7136, 46.6753),
    "Jeddah": (21.5433, 39.1728),
    "Mecca": (21.3891, 39.8579),
    "Medina": (24.5247, 39.5692),
    "Dammam": (26.4207, 50.0888),
}


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
        values["_source_kind"] = "historical"
        values["_source_label"] = "Project dataset آ· historical observation"
        values["_air_quality_available"] = True
        return values


class OpenMeteoClient:
    """Retrieve model-ready weather and air-quality features by city and date."""

    WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    FORECAST_DAYS = 16
    HOURLY_WEATHER = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "cloud_cover",
        "precipitation",
        "shortwave_radiation",
        "sunshine_duration",
    ]
    HOURLY_AIR = [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
    ]

    @staticmethod
    def _request(url: str, params: Dict) -> Dict:
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{url}?{query}",
            headers={"User-Agent": "Solar-Decision-Intelligence/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _numbers(values) -> List[float]:
        return [
            float(value)
            for value in (values or [])
            if value is not None and not pd.isna(value)
        ]

    @classmethod
    def _mean(cls, values) -> Optional[float]:
        numbers = cls._numbers(values)
        return float(np.mean(numbers)) if numbers else None

    @classmethod
    def _sum(cls, values) -> Optional[float]:
        numbers = cls._numbers(values)
        return float(np.sum(numbers)) if numbers else None

    @property
    def latest_forecast_date(self) -> date:
        return datetime.now().date() + timedelta(days=self.FORECAST_DAYS - 1)

    @staticmethod
    def _display_date(value: date) -> str:
        return value.strftime("%B %d, %Y").replace(" 0", " ")

    @lru_cache(maxsize=256)
    def get_features(self, city: str, date_str: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Return API features or a conversational reason they are unavailable."""
        coordinates = CITY_COORDINATES.get(city)
        target = pd.to_datetime(date_str, errors="coerce")
        if coordinates is None or pd.isna(target):
            return None, "I could not resolve that location and date."

        target_date = target.date()
        today = datetime.now().date()
        if target_date > self.latest_forecast_date:
            return (
                None,
                (
                    f"I can use live forecast inputs through "
                    f"{self._display_date(self.latest_forecast_date)}. "
                    f"{self._display_date(target_date)} is too far ahead for "
                    "a reliable weather-based prediction."
                ),
            )

        is_forecast = target_date >= today
        use_forecast_api = target_date >= today - timedelta(days=5)
        weather_url = (
            self.WEATHER_FORECAST_URL
            if use_forecast_api
            else self.WEATHER_ARCHIVE_URL
        )
        common = {
            "latitude": coordinates[0],
            "longitude": coordinates[1],
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "Asia/Riyadh",
        }

        try:
            weather = self._request(
                weather_url,
                {**common, "hourly": ",".join(self.HOURLY_WEATHER)},
            ).get("hourly", {})
        except (
            json.JSONDecodeError,
            OSError,
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ):
            return (
                None,
                (
                    "I could not reach the live weather service for that date. "
                    "Please try the same request again in a moment."
                ),
            )

        values = {
            "temperature_2m_mean": self._mean(weather.get("temperature_2m")),
            "relative_humidity_2m_mean": self._mean(
                weather.get("relative_humidity_2m")
            ),
            "surface_pressure_mean": self._mean(weather.get("surface_pressure")),
            "wind_speed_10m_mean": self._mean(weather.get("wind_speed_10m")),
            "cloud_cover_mean": self._mean(weather.get("cloud_cover")),
            "precipitation_sum": self._sum(weather.get("precipitation")),
            # Hourly shortwave radiation is W/mآ². Summing 1-hour values and
            # multiplying by 0.0036 converts the result to MJ/mآ².
            "shortwave_radiation_sum": (
                self._sum(weather.get("shortwave_radiation")) * 0.0036
                if self._sum(weather.get("shortwave_radiation")) is not None
                else None
            ),
            "sunshine_duration": self._sum(weather.get("sunshine_duration")),
        }
        if values["temperature_2m_mean"] is None:
            return None, "The weather service returned no usable data for that day."

        air_available = False
        try:
            air = self._request(
                self.AIR_QUALITY_URL,
                {**common, "hourly": ",".join(self.HOURLY_AIR)},
            ).get("hourly", {})
            for feature in self.HOURLY_AIR:
                feature_value = self._mean(air.get(feature))
                if feature_value is not None:
                    values[feature] = feature_value
                    air_available = True
        except (
            json.JSONDecodeError,
            OSError,
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ):
            # Weather-only predictions remain useful. Missing pollution
            # features are filled from training medians by ModelTrainer.
            pass

        source_kind = "forecast" if is_forecast else "historical"
        values.update(
            {
                "Date": date_str,
                "City": city,
                "_source_kind": source_kind,
                "_source_label": (
                    "Open-Meteo آ· forecast inputs"
                    if is_forecast
                    else "Open-Meteo آ· historical weather"
                ),
                "_air_quality_available": air_available,
            }
        )
        return values, None


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
        self.feature_medians: Dict[str, float] = {}
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
        medians = training[self.feature_names].median(numeric_only=True)
        self.feature_medians = {
            name: float(medians.get(name, 0) or 0) for name in self.feature_names
        }
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
                    name: float(
                        feature_dict.get(name)
                        if feature_dict.get(name) is not None
                        else self.feature_medians.get(name, 0)
                    )
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
        source_kind = data.get("_source_kind", "historical")
        evidence_label = (
            "forecast inputs" if source_kind == "forecast" else "historical inputs"
        )
        prompt = (
            f"User question: {user_query}\n"
            f"Detected topics: {', '.join(intents)}\n"
            f"Location and date: {city}, {date_str}\n"
            f"Evidence type: {evidence_label}\n"
            f"Weather and air-quality features: {json.dumps(observed, default=str)}\n"
            f"Model predictions: {json.dumps(predictions, default=str)}\n"
            "Retrieved domain guidance:\n- "
            + "\n- ".join(interpretations)
            + "\n\nAnswer the user's exact question first, then connect the relevant "
            "weather observations to the solar-energy and air-quality predictions. "
            "If the user asks whether the location is suitable for solar, discuss both "
            "supporting and limiting factors. Include the predicted daily solar energy "
            "when relevant. Use no more than 220 words, mention important uncertainty, "
            "and give one practical recommendation. Use only the supplied facts. "
            "Accurately distinguish forecast inputs from historical observations."
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
        self.live_data = OpenMeteoClient()
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

    def process_query(
        self,
        user_query: str,
        context: Optional[Dict] = None,
    ) -> Dict:
        city, date_str = self._extract_location_date(user_query)
        intents = self._detect_intents(user_query)
        context = context or {}
        city = city or context.get("city")
        date_str = date_str or context.get("date")
        if intents == ["combined conditions"] and context.get("intents"):
            intents = context["intents"]
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
                error=(
                    "Which location would you like me to analyse? I currently cover "
                    f"{', '.join(self.data_loader.cities)}."
                ),
            )
            return result

        if not date_str:
            result.update(
                status="error",
                error=(
                    f"What date should I use for {city}? You can write it naturally, "
                    "for example â€œFebruary 2, 2026â€‌, â€œ2 February 2026â€‌, "
                    "â€œtomorrowâ€‌, or â€œlast Fridayâ€‌."
                ),
            )
            return result

        data = self.data_loader.get_data_for_location_date(city, date_str)
        if not data:
            data, live_error = self.live_data.get_features(city, date_str)
            if not data:
                result.update(status="error", error=live_error)
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
            source_label=data.get("_source_label", "Model inputs"),
            source_kind=data.get("_source_kind", "historical"),
            air_quality_available=data.get("_air_quality_available", True),
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
            weather_parts.append(f"mean temperature {temperature:.1f} آ°C")
        if humidity is not None:
            weather_parts.append(f"humidity {humidity:.0f}%")
        if wind is not None:
            weather_parts.append(f"wind {wind:.1f} km/h")
        if cloud is not None:
            weather_parts.append(f"cloud cover {cloud:.0f}%")
        if rain is not None:
            weather_parts.append(f"precipitation {rain:.1f} mm")

        source_kind = data.get("_source_kind", "historical")
        date_label = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d, %Y")
        date_label = date_label.replace(" 0", " ")
        sentences = [f"For {city} on {date_label}:"]
        if "weather" in intents or "combined conditions" in intents:
            weather_noun = (
                "Forecast conditions are"
                if source_kind == "forecast"
                else "Weather conditions were"
            )
            sentences.append(f"{weather_noun} " + ", ".join(weather_parts) + ".")
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

        date_str = self._extract_date(query)
        return city, date_str

    @staticmethod
    def _valid_date(year: int, month: int, day: int) -> Optional[str]:
        try:
            return date(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    @classmethod
    def _extract_date(cls, query: str) -> Optional[str]:
        """Understand common conversational date expressions without dependencies."""
        normalized = query.lower().replace("â€™", "'")
        today = datetime.now().date()

        relative_days = {
            "day after tomorrow": 2,
            "day before yesterday": -2,
            "tomorrow": 1,
            "yesterday": -1,
            "today": 0,
        }
        for phrase, offset in relative_days.items():
            if phrase in normalized:
                return (today + timedelta(days=offset)).strftime("%Y-%m-%d")

        relative_match = re.search(
            r"\b(?:in\s+(\d+)\s+days?|(\d+)\s+days?\s+ago)\b",
            normalized,
        )
        if relative_match:
            amount = int(relative_match.group(1) or relative_match.group(2))
            if relative_match.group(2):
                amount *= -1
            return (today + timedelta(days=amount)).strftime("%Y-%m-%d")

        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        weekday_match = re.search(
            r"\b(next|last|this)\s+(" + "|".join(weekdays) + r")\b",
            normalized,
        )
        if weekday_match:
            direction, weekday = weekday_match.groups()
            target_weekday = weekdays[weekday]
            if direction == "last":
                delta = -((today.weekday() - target_weekday) % 7 or 7)
            elif direction == "next":
                delta = (target_weekday - today.weekday()) % 7 or 7
            else:
                delta = (target_weekday - today.weekday()) % 7
            return (today + timedelta(days=delta)).strftime("%Y-%m-%d")

        iso_match = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", normalized)
        if iso_match:
            return cls._valid_date(*(int(part) for part in iso_match.groups()))

        numeric_match = re.search(
            r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b",
            normalized,
        )
        if numeric_match:
            day, month, year = (int(part) for part in numeric_match.groups())
            return cls._valid_date(year, month, day)

        months = {
            "january": 1,
            "jan": 1,
            "february": 2,
            "feb": 2,
            "march": 3,
            "mar": 3,
            "april": 4,
            "apr": 4,
            "may": 5,
            "june": 6,
            "jun": 6,
            "july": 7,
            "jul": 7,
            "august": 8,
            "aug": 8,
            "september": 9,
            "sep": 9,
            "sept": 9,
            "october": 10,
            "oct": 10,
            "november": 11,
            "nov": 11,
            "december": 12,
            "dec": 12,
        }
        month_names = "|".join(sorted(months, key=len, reverse=True))
        month_first = re.search(
            rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(\d{{4}})\b",
            normalized,
        )
        if month_first:
            month_name, day, year = month_first.groups()
            return cls._valid_date(int(year), months[month_name], int(day))

        day_first = re.search(
            rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?"
            rf"({month_names})(?:,)?\s+(\d{{4}})\b",
            normalized,
        )
        if day_first:
            day, month_name, year = day_first.groups()
            return cls._valid_date(int(year), months[month_name], int(day))
        return None


def main():
    rag = SolarRAG()
    rag.setup()
    print(rag.process_query("Solar conditions in Riyadh on 2024-01-15"))


if __name__ == "__main__":
    main()
