"""Short RAG variant that loads persisted Project_Solar notebook models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

from rag import CITY_COORDINATES, OpenMeteoClient, SolarRAG


MODEL_BUNDLE = Path(__file__).resolve().parent / "models" / "project_solar_models.joblib"


class NotebookOpenMeteoClient(OpenMeteoClient):
    """Include current AQI because the notebook models use it as an input."""

    HOURLY_AIR = [*OpenMeteoClient.HOURLY_AIR, "us_aqi"]


class SavedNotebookPredictor:
    """Load the notebook model bundle once and perform inference only."""

    def __init__(self, model_path: Path = MODEL_BUNDLE):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved models not found at {model_path}. "
                "Run `python export_notebook_models.py` once."
            )
        self.bundle = joblib.load(model_path)

    @staticmethod
    def _number(value, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    def _inputs(self, data: Dict, section: str) -> pd.DataFrame:
        spec = self.bundle[section]
        row = dict(spec["medians"])
        for feature in spec["features"]:
            if data.get(feature) is not None:
                row[feature] = self._number(data[feature], row.get(feature, 0))

        city = str(data.get("City", "Dammam")).title()
        date = pd.to_datetime(data.get("Date"), errors="coerce")
        if pd.isna(date):
            date = pd.Timestamp(datetime.now().date())

        latitude, longitude = CITY_COORDINATES.get(city, CITY_COORDINATES["Dammam"])
        row["Latitude"] = self._number(data.get("Latitude"), latitude)
        row["Longitude"] = self._number(data.get("Longitude"), longitude)
        row["Month"] = date.month
        row["DayOfYear"] = date.dayofyear

        for feature in spec["features"]:
            if feature.startswith(("City_", "Panel Type_", "Mount Type_", "Weekday_")):
                row[feature] = 0
        for active in (
            f"City_{city}",
            "Mount Type_Rooftop",
            f"Weekday_{date.day_name()}",
        ):
            if active in row:
                row[active] = 1

        return pd.DataFrame([row], columns=spec["features"])

    def predict(self, data: Dict) -> Dict:
        solar = self.bundle["solar"]
        aqi = self.bundle["aqi"]
        solar_value = float(solar["model"].predict(self._inputs(data, "solar"))[0])
        aqi_inputs = self._inputs(data, "aqi")
        aqi_value = float(aqi["value_model"].predict(aqi_inputs)[0])
        risk_code = int(aqi["risk_model"].predict(aqi_inputs)[0])
        return {
            "solar_output_kwh": max(0.0, solar_value),
            "aqi_value": max(0.0, aqi_value),
            "aqi_risk_level": aqi["risk_labels"].get(risk_code, "Unknown"),
            "aqi_horizon": aqi["horizon"],
            "model_source": "Project_Solar.ipynb saved models",
        }


class SavedModelSolarRAG(SolarRAG):
    """Keep the existing RAG workflow but replace training with saved-model calls."""

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        model_path: Path = MODEL_BUNDLE,
        **kwargs,
    ):
        super().__init__(dataset_path=dataset_path, **kwargs)
        self.trainer = SavedNotebookPredictor(model_path)
        self.live_data = NotebookOpenMeteoClient()
        limits = self.trainer.bundle["solar"]["output_thresholds"]
        self.kb.knowledge_items[:3] = [
            f"Solar output above {limits['high']:.1f} kWh is high for this project.",
            (
                f"Solar output between {limits['low']:.1f} and "
                f"{limits['high']:.1f} kWh is moderate for this project."
            ),
            f"Solar output below {limits['low']:.1f} kWh is low for this project.",
        ]
        self.kb.embeddings = self.kb.vectorizer.fit_transform(self.kb.knowledge_items)


def main() -> None:
    rag = SavedModelSolarRAG()
    rag.setup()
    print(rag.process_query("Solar outlook for Riyadh on February 2, 2024"))


if __name__ == "__main__":
    main()
