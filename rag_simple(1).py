"""Simplified SolarIQ RAG backend compatible with the existing Streamlit chat."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = APP_DIR / "data" / "solar_dataset.csv"
DEFAULT_MODELS = APP_DIR / "models" / "solar_rag_models.joblib"

CITIES = ["Jeddah", "Mecca", "Medina", "Riyadh", "Dammam"]
CITY_ALIASES = {"makkah": "Mecca", "madinah": "Medina"}
CITY_COORDINATES = {
    "Riyadh": (24.7136, 46.6753),
    "Jeddah": (21.5433, 39.1728),
    "Mecca": (21.3891, 39.8579),
    "Medina": (24.5247, 39.5692),
    "Dammam": (26.4207, 50.0888),
}
CITY_COLUMNS = {
    "Jeddah": "City_Jeddah",
    "Mecca": "City_Mecca",
    "Medina": "City_Medina",
    "Riyadh": "City_Riyadh",
}

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
KNOWLEDGE = [
    "Higher shortwave radiation and longer sunshine duration usually increase solar output.",
    "Cloud cover and precipitation reduce the radiation reaching photovoltaic panels.",
    "Dust and particulate matter can settle on panels and reduce their efficiency.",
    "High temperatures can reduce photovoltaic conversion efficiency.",
    "Wind can cool solar panels, although very high wind requires operational precautions.",
    "AQI below 50 is good, 50 to 100 is moderate, and above 100 is unhealthy.",
    "Solar suitability depends on radiation, clouds, heat, dust, shading, tilt, and system efficiency.",
    "A model prediction is an estimate for configurations represented in the training data.",
]

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
API_TIMEOUT = 10


def _dataset_candidates(path: Path) -> List[Path]:
    candidates = [path]
    if path.suffix != ".gz":
        candidates.append(path.with_name(f"{path.name}.gz"))
    return candidates


def _load_dataset(path: Path) -> Tuple[pd.DataFrame, Path]:
    failures = []
    for candidate in _dataset_candidates(path):
        if not candidate.exists():
            failures.append(f"{candidate.name}: missing")
            continue
        try:
            frame = pd.read_csv(candidate)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            failures.append(f"{candidate.name}: {exc}")
            continue
        if "Date" not in frame:
            failures.append(f"{candidate.name}: missing Date")
            continue
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).copy()
        if not frame.empty:
            return frame, candidate
    raise ValueError("Unable to load the solar dataset. " + "; ".join(failures))


def _city_mask(frame: pd.DataFrame, city: str) -> pd.Series:
    if "City" in frame:
        return frame["City"].astype(str).str.lower().eq(city.lower())
    if city in CITY_COLUMNS and CITY_COLUMNS[city] in frame:
        return frame[CITY_COLUMNS[city]].fillna(0).astype(int).eq(1)
    encoded = [column for column in CITY_COLUMNS.values() if column in frame]
    if city == "Dammam" and encoded:
        return frame[encoded].fillna(0).sum(axis=1).eq(0)
    return pd.Series(False, index=frame.index)


def _row_city(row: pd.Series) -> str:
    if "City" in row and pd.notna(row["City"]):
        return str(row["City"])
    for city, column in CITY_COLUMNS.items():
        if column in row and pd.to_numeric(row[column], errors="coerce") == 1:
            return city
    return "Dammam"


def _historical_day(
    frame: pd.DataFrame,
    city: str,
    date_str: str,
) -> Optional[Dict]:
    target = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(target):
        return None
    rows = frame[
        _city_mask(frame, city) & frame["Date"].dt.date.eq(target.date())
    ]
    if rows.empty:
        return None
    values = rows.select_dtypes(include=[np.number]).mean().to_dict()
    values.update(
        Date=date_str,
        City=city,
        _source_kind="historical",
        _source_label="Project dataset · historical observation",
        _air_quality_available=True,
    )
    return values


def _request(url: str, params: Dict) -> Dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "SolarIQ/1.0"},
    )
    with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _numbers(values) -> List[float]:
    return [
        float(value)
        for value in (values or [])
        if value is not None and not pd.isna(value)
    ]


def _mean(values) -> Optional[float]:
    numbers = _numbers(values)
    return float(np.mean(numbers)) if numbers else None


def _sum(values) -> Optional[float]:
    numbers = _numbers(values)
    return float(np.sum(numbers)) if numbers else None


@lru_cache(maxsize=512)
def _api_features(city: str, date_str: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Fetch and cache weather and air-quality data for one city and date."""
    coordinates = CITY_COORDINATES.get(city)
    target = pd.to_datetime(date_str, errors="coerce")
    if coordinates is None or pd.isna(target):
        return None, "I could not resolve that location and date."

    today = datetime.now().date()
    target_date = target.date()
    if target_date > today + timedelta(days=15):
        return None, "That date is too far ahead for a reliable weather forecast."

    is_forecast = target_date >= today
    weather_url = WEATHER_URL if target_date >= today - timedelta(days=5) else ARCHIVE_URL
    common = {
        "latitude": coordinates[0],
        "longitude": coordinates[1],
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "Asia/Riyadh",
    }
    weather_params = {
        **common,
        "hourly": (
            "temperature_2m,relative_humidity_2m,surface_pressure,"
            "wind_speed_10m,cloud_cover,precipitation,"
            "shortwave_radiation,sunshine_duration"
        ),
    }
    air_params = {**common, "hourly": ",".join(AIR_FEATURES)}
    errors = (
        json.JSONDecodeError,
        OSError,
        TimeoutError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        weather_future = executor.submit(_request, weather_url, weather_params)
        air_future = executor.submit(_request, AIR_URL, air_params)
        try:
            weather = weather_future.result().get("hourly", {})
        except errors:
            weather = {}
        try:
            air = air_future.result().get("hourly", {})
        except errors:
            air = {}

    if not weather:
        return None, "The weather service is unavailable. Please try again."

    radiation = _sum(weather.get("shortwave_radiation"))
    values = {
        "temperature_2m_mean": _mean(weather.get("temperature_2m")),
        "relative_humidity_2m_mean": _mean(weather.get("relative_humidity_2m")),
        "surface_pressure_mean": _mean(weather.get("surface_pressure")),
        "wind_speed_10m_mean": _mean(weather.get("wind_speed_10m")),
        "cloud_cover_mean": _mean(weather.get("cloud_cover")),
        "precipitation_sum": _sum(weather.get("precipitation")),
        "shortwave_radiation_sum": radiation * 0.0036 if radiation is not None else None,
        "sunshine_duration": _sum(weather.get("sunshine_duration")),
    }
    air_available = False
    for feature in AIR_FEATURES:
        value = _mean(air.get(feature))
        if value is not None:
            values[feature] = value
            air_available = True
    values.update(
        Date=date_str,
        City=city,
        _source_kind="forecast" if is_forecast else "historical",
        _source_label=(
            "Open-Meteo · forecast inputs"
            if is_forecast
            else "Open-Meteo · historical inputs"
        ),
        _air_quality_available=air_available,
    )
    return values, None


def _signature(path: Path) -> Dict:
    return {"name": path.name, "size": path.stat().st_size}


def _load_or_train_models(
    frame: pd.DataFrame,
    dataset_path: Path,
    model_path: Path,
) -> Dict:
    """Load the joblib bundle or train and save it once."""
    if model_path.exists():
        try:
            bundle = joblib.load(model_path)
            if bundle.get("dataset_signature") == _signature(dataset_path):
                return bundle
        except (EOFError, KeyError, OSError, TypeError, ValueError):
            pass

    features = [
        name for name in WEATHER_FEATURES + AIR_FEATURES if name in frame
    ]
    required = {"Estimated Daily Output (kWh)", "us_aqi"}
    if not features or not required.issubset(frame.columns):
        raise ValueError("The dataset does not contain the required model columns.")

    training = frame[features + list(required)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    medians = training[features].median(numeric_only=True)
    feature_medians = {
        name: float(medians.get(name, 0) or 0) for name in features
    }
    training = training.fillna(training.median(numeric_only=True)).dropna()
    if len(training) > 6000:
        training = training.sample(6000, random_state=42)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(training[features])
    y_solar = training["Estimated Daily Output (kWh)"]
    y_aqi = training["us_aqi"]
    y_class = pd.cut(
        y_aqi,
        bins=[-np.inf, 50, 100, np.inf],
        labels=[0, 1, 2],
    ).astype(int)

    bundle = {
        "bundle_version": 1,
        "dataset_signature": _signature(dataset_path),
        "feature_names": features,
        "feature_medians": feature_medians,
        "scaler": scaler,
        "model_solar_reg": RandomForestRegressor(
            n_estimators=40,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ).fit(x_scaled, y_solar),
        "model_aqi_reg": LinearRegression().fit(x_scaled, y_aqi),
        "model_aqi_class": RandomForestClassifier(
            n_estimators=40,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ).fit(x_scaled, y_class),
    }
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix(".tmp")
        joblib.dump(bundle, temporary)
        os.replace(temporary, model_path)
    except OSError:
        pass
    return bundle


def _predict(bundle: Dict, data: Dict) -> Dict:
    features = bundle["feature_names"]
    medians = bundle["feature_medians"]
    row = {
        name: float(data[name]) if data.get(name) is not None else medians[name]
        for name in features
    }
    scaled = bundle["scaler"].transform(pd.DataFrame([row], columns=features))
    risk_names = {0: "Good", 1: "Moderate", 2: "Unhealthy"}
    risk_class = int(bundle["model_aqi_class"].predict(scaled)[0])
    return {
        "solar_output_kwh": float(
            max(0, bundle["model_solar_reg"].predict(scaled)[0])
        ),
        "aqi_value": float(max(0, bundle["model_aqi_reg"].predict(scaled)[0])),
        "aqi_risk_level": risk_names.get(risk_class, "Unknown"),
    }


def _find_city(query: str) -> Optional[str]:
    lower = query.lower()
    for city in CITIES:
        if city.lower() in lower:
            return city
    for alias, city in CITY_ALIASES.items():
        if alias in lower:
            return city
    return None


def _valid_date(year: int, month: int, day: int) -> Optional[str]:
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _find_date(query: str) -> Optional[str]:
    lower = query.lower()
    today = datetime.now().date()
    if "tomorrow" in lower:
        return (today + timedelta(days=1)).isoformat()
    if "yesterday" in lower:
        return (today - timedelta(days=1)).isoformat()
    if re.search(r"\btoday\b", lower):
        return today.isoformat()

    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", lower)
    if iso:
        year, month, day = map(int, iso.groups())
        return _valid_date(year, month, day)

    numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", lower)
    if numeric:
        day, month, year = map(int, numeric.groups())
        return _valid_date(year, month, day)

    months = {
        name.lower(): number
        for number, name in enumerate(
            [
                "",
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        )
        if name
    }
    month_names = "|".join(months)
    normalized = re.sub(r"[,]+", " ", lower)
    month_first = re.search(
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+(\d{{4}})\b",
        normalized,
    )
    if month_first:
        month, day, year = month_first.groups()
        return _valid_date(int(year), months[month], int(day))
    day_first = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})\s+(\d{{4}})\b",
        normalized,
    )
    if day_first:
        day, month, year = day_first.groups()
        return _valid_date(int(year), months[month], int(day))
    return None


def _operation(query: str) -> Optional[str]:
    lower = query.lower()
    if any(word in lower for word in ("highest", "maximum", "max ", "best")):
        return "maximum"
    if any(word in lower for word in ("lowest", "minimum", "min ", "worst")):
        return "minimum"
    if any(word in lower for word in ("average", "mean")):
        return "average"
    return None


def _clean(data: Dict) -> Dict:
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, np.generic):
            value = value.item()
        if not (isinstance(value, float) and np.isnan(value)):
            cleaned[key] = value
    return cleaned


class SolarRAG:
    """Small end-to-end solar RAG with the same interface as the full version."""

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        ollama_api_key: Optional[str] = None,
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ):
        del ollama_api_key, ollama_host, ollama_model
        self.dataset_path = Path(dataset_path or DEFAULT_DATASET)
        self.frame: Optional[pd.DataFrame] = None
        self.models: Optional[Dict] = None
        self.ready = False
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.knowledge_vectors = self.vectorizer.fit_transform(KNOWLEDGE)

    def setup(self) -> bool:
        self.frame, self.dataset_path = _load_dataset(self.dataset_path)
        self.ready = True
        return True

    def _retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.knowledge_vectors)[0]
        indices = scores.argsort()[::-1][:top_k]
        return [KNOWLEDGE[index] for index in indices]

    def _models(self) -> Dict:
        if self.models is None:
            assert self.frame is not None
            self.models = _load_or_train_models(
                self.frame,
                self.dataset_path,
                DEFAULT_MODELS,
            )
        return self.models

    def _aggregate(self, query: str, city: Optional[str], operation: str) -> Dict:
        assert self.frame is not None
        target = "Estimated Daily Output (kWh)"
        rows = self.frame[_city_mask(self.frame, city)] if city else self.frame
        values = pd.to_numeric(rows.get(target), errors="coerce").dropna()
        if values.empty:
            return {"status": "error", "error": "No matching energy records were found."}

        if operation == "maximum":
            row = rows.loc[values.idxmax()]
            output = float(values.max())
            date_label = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
            result_city = city or _row_city(row)
        elif operation == "minimum":
            row = rows.loc[values.idxmin()]
            output = float(values.min())
            date_label = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
            result_city = city or _row_city(row)
        else:
            row = rows.select_dtypes(include=[np.number]).mean()
            output = float(values.mean())
            date_label = "Full historical period"
            result_city = city or "All markets"

        raw_aqi = pd.to_numeric(row.get("us_aqi"), errors="coerce")
        aqi = 0.0 if pd.isna(raw_aqi) else float(raw_aqi)
        risk = "Good" if aqi < 50 else "Moderate" if aqi <= 100 else "Unhealthy"
        interpretations = self._retrieve(f"{query} {output:.1f}", top_k=3)
        answer = (
            f"The {operation} estimated solar output for {result_city} was "
            f"{output:.1f} kWh"
            + (
                f" on {date_label}."
                if operation != "average"
                else " across the historical dataset."
            )
            + f" The associated AQI value was approximately {aqi:.0f} ({risk}). "
            + interpretations[0]
        )
        return {
            "query": query,
            "status": "success",
            "data": _clean(row.to_dict()),
            "predictions": {
                "solar_output_kwh": output,
                "aqi_value": aqi,
                "aqi_risk_level": risk,
            },
            "interpretations": interpretations,
            "intents": ["historical analysis"],
            "city": result_city,
            "date": date_label,
            "llm_response": answer,
            "llm_status": "not_used",
            "llm_error": None,
            "source_label": "Project dataset · historical analysis",
            "source_kind": "historical",
            "air_quality_available": "us_aqi" in row,
        }

    def process_query(
        self,
        user_query: str,
        context: Optional[Dict] = None,
    ) -> Dict:
        if not self.ready:
            self.setup()
        context = context or {}
        city = _find_city(user_query) or context.get("city")
        operation = _operation(user_query)
        if operation:
            return self._aggregate(user_query, city, operation)

        date_str = _find_date(user_query) or context.get("date")
        if not city:
            return {
                "status": "error",
                "error": f"Which city should I analyse? I cover {', '.join(CITIES)}.",
            }
        if not date_str:
            return {
                "status": "error",
                "error": f"What date should I use for {city}?",
            }

        assert self.frame is not None
        data = _historical_day(self.frame, city, date_str)
        api_error = None
        if data is None:
            data, api_error = _api_features(city, date_str)
        if data is None:
            return {"status": "error", "error": api_error}

        predictions = _predict(self._models(), data)
        interpretations = self._retrieve(
            f"{user_query} {predictions['solar_output_kwh']:.1f} "
            f"{predictions['aqi_value']:.1f}",
            top_k=3,
        )
        date_label = datetime.strptime(date_str, "%Y-%m-%d").strftime(
            "%B %d, %Y"
        ).replace(" 0", " ")
        answer = (
            f"For {city} on {date_label}, the solar model estimates "
            f"{predictions['solar_output_kwh']:.1f} kWh of daily output. "
            f"The AQI estimate is {predictions['aqi_value']:.0f} "
            f"({predictions['aqi_risk_level']}). {interpretations[0]}"
        )
        return {
            "query": user_query,
            "status": "success",
            "data": _clean(data),
            "predictions": predictions,
            "interpretations": interpretations,
            "intents": ["solar and air-quality analysis"],
            "city": city,
            "date": date_str,
            "llm_response": answer,
            "llm_status": "not_used",
            "llm_error": None,
            "source_label": data.get("_source_label", "Model inputs"),
            "source_kind": data.get("_source_kind", "historical"),
            "air_quality_available": data.get("_air_quality_available", True),
        }
