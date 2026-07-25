"""Train the selected Project_Solar notebook models and save one reusable bundle."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "solar_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_BUNDLE = MODEL_DIR / "project_solar_models.joblib"

SOLAR_FEATURES = [
    "Latitude",
    "Longitude",
    "Tilt (°)",
    "Panel Efficiency (%)",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
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
    "us_aqi",
    "City_Jeddah",
    "City_Mecca",
    "City_Medina",
    "City_Riyadh",
    "Panel Type_Polycrystalline",
    "Panel Type_Thin-Film",
    "Mount Type_Rooftop",
    "Weekday_Monday",
    "Weekday_Saturday",
    "Weekday_Sunday",
    "Weekday_Thursday",
    "Weekday_Tuesday",
    "Weekday_Wednesday",
]

AQI_FEATURES = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
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
    "us_aqi",
    "Month",
    "DayOfYear",
    "City_Jeddah",
    "City_Mecca",
    "City_Medina",
    "City_Riyadh",
]


def numeric_frame(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Return numeric model inputs filled with reusable training medians."""
    values = frame[features].apply(pd.to_numeric, errors="coerce")
    medians = values.median(numeric_only=True).fillna(0).to_dict()
    return values.fillna(medians), {key: float(value) for key, value in medians.items()}


def train_and_export() -> Path:
    """Reproduce the notebook's selected models and persist them for inference."""
    frame = pd.read_csv(DATASET)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).copy()

    solar_x, solar_medians = numeric_frame(frame, SOLAR_FEATURES)
    solar_y = pd.to_numeric(
        frame["Estimated Daily Output (kWh)"], errors="coerce"
    )
    solar_rows = solar_y.notna()
    solar_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    ).fit(solar_x.loc[solar_rows], solar_y.loc[solar_rows])

    air = (
        frame.drop_duplicates(subset=["Date", "Latitude", "Longitude"])
        .sort_values(["Latitude", "Longitude", "Date"])
        .copy()
    )
    air["Next_Day_AQI"] = air.groupby(["Latitude", "Longitude"])["us_aqi"].shift(-1)
    air["Month"] = air["Date"].dt.month
    air["DayOfYear"] = air["Date"].dt.dayofyear
    air = air.dropna(subset=["Next_Day_AQI"])

    aqi_x, aqi_medians = numeric_frame(air, AQI_FEATURES)
    aqi_y = pd.to_numeric(air["Next_Day_AQI"], errors="coerce")
    aqi_rows = aqi_y.notna()
    aqi_x = aqi_x.loc[aqi_rows]
    aqi_y = aqi_y.loc[aqi_rows]

    aqi_value_model = LinearRegression().fit(aqi_x, aqi_y)
    aqi_risk_y = pd.cut(
        aqi_y,
        bins=[-np.inf, 100, 200, np.inf],
        labels=[0, 1, 2],
    ).astype(int)
    aqi_risk_model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
    ).fit(aqi_x, aqi_risk_y)

    bundle = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_notebook": "Project_Solar.ipynb",
        "solar": {
            "model": solar_model,
            "features": SOLAR_FEATURES,
            "medians": solar_medians,
            "output_thresholds": {
                "low": float(solar_y.quantile(1 / 3)),
                "high": float(solar_y.quantile(2 / 3)),
            },
            "training_rows": int(solar_rows.sum()),
        },
        "aqi": {
            "value_model": aqi_value_model,
            "risk_model": aqi_risk_model,
            "features": AQI_FEATURES,
            "medians": aqi_medians,
            "risk_labels": {
                0: "Acceptable",
                1: "Elevated Risk",
                2: "Severe Risk",
            },
            "horizon": "next day",
            "training_rows": int(aqi_rows.sum()),
        },
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_BUNDLE, compress=3)
    return MODEL_BUNDLE


if __name__ == "__main__":
    output = train_and_export()
    print(f"Saved notebook model bundle to {output}")
