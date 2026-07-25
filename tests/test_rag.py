"""Regression tests for the conversational Solar RAG workflow."""

import json
import unittest
from unittest.mock import patch

from rag import OllamaExplainer, SolarRAG


class FakeOllamaResponse:
    """Minimal context-manager response for a successful Ollama API call."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(
            {"message": {"content": "Grounded Ollama explanation."}}
        ).encode("utf-8")


class SolarRAGConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = SolarRAG()
        cls.rag.setup()

    def test_supported_question_types(self):
        questions = {
            "What was the weather in Riyadh on 2024-02-02?": "weather",
            "How much solar energy could I get in Jeddah on 2024-06-15?": (
                "solar energy"
            ),
            "How was the air quality in Dammam on 2024-03-10?": "air quality",
            "Was Medina suitable for solar generation on 2024-09-01?": (
                "solar suitability"
            ),
        }

        for question, expected_intent in questions.items():
            with self.subTest(question=question):
                result = self.rag.process_query(question)
                self.assertEqual(result["status"], "success")
                self.assertIn(expected_intent, result["intents"])
                self.assertGreater(result["predictions"]["solar_output_kwh"], 0)
                self.assertIn("temperature_2m_mean", result["data"])
                self.assertTrue(result["llm_response"])

    def test_question_requires_a_historical_date(self):
        result = self.rag.process_query("How much solar energy in Riyadh?")

        self.assertEqual(result["status"], "error")
        self.assertIn("What date", result["error"])

    def test_natural_language_date_is_understood(self):
        city, date_str = self.rag._extract_location_date(
            "What is the Riyadh outlook for February 2nd, 2026?"
        )

        self.assertEqual(city, "Riyadh")
        self.assertEqual(date_str, "2026-02-02")

    def test_follow_up_reuses_conversation_context(self):
        result = self.rag.process_query(
            "What about Jeddah?",
            context={
                "city": "Riyadh",
                "date": "2024-02-02",
                "intents": ["solar energy"],
            },
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["city"], "Jeddah")
        self.assertEqual(result["date"], "2024-02-02")
        self.assertEqual(result["intents"], ["solar energy"])

    def test_out_of_dataset_date_uses_live_features(self):
        api_features = {
            "temperature_2m_mean": 22.0,
            "relative_humidity_2m_mean": 35.0,
            "surface_pressure_mean": 950.0,
            "wind_speed_10m_mean": 11.0,
            "cloud_cover_mean": 18.0,
            "precipitation_sum": 0.0,
            "shortwave_radiation_sum": 19.0,
            "sunshine_duration": 36000.0,
            "pm10": 44.0,
            "pm2_5": 18.0,
            "carbon_monoxide": 160.0,
            "nitrogen_dioxide": 13.0,
            "ozone": 70.0,
            "sulphur_dioxide": 8.0,
            "Date": "2026-02-02",
            "City": "Riyadh",
            "_source_kind": "historical",
            "_source_label": "Open-Meteo · historical weather",
            "_air_quality_available": True,
        }
        with patch.object(
            self.rag.live_data,
            "get_features",
            return_value=(api_features, None),
        ) as get_features:
            result = self.rag.process_query(
                "Solar outlook for Riyadh on February 2, 2026"
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["source_kind"], "historical")
        self.assertGreater(result["predictions"]["solar_output_kwh"], 0)
        get_features.assert_called_once_with("Riyadh", "2026-02-02")

    def test_makkah_alias_maps_to_mecca(self):
        result = self.rag.process_query(
            "Was Makkah suitable for solar energy on 2024/04/15?"
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["city"], "Mecca")
        self.assertEqual(result["date"], "2024-04-15")

    @patch("urllib.request.urlopen", return_value=FakeOllamaResponse())
    def test_ollama_explanation_is_used_when_configured(self, _mock_urlopen):
        explainer = OllamaExplainer(api_key="test-key")

        explanation, status, error = explainer.explain(
            user_query="How much solar energy?",
            city="Riyadh",
            date_str="2024-02-02",
            data={"temperature_2m_mean": 20.0},
            predictions={
                "solar_output_kwh": 100.0,
                "aqi_value": 70.0,
                "aqi_risk_level": "Moderate",
            },
            interpretations=["Moderate solar conditions."],
            intents=["solar energy"],
        )

        self.assertEqual(explanation, "Grounded Ollama explanation.")
        self.assertEqual(status, "ollama_cloud")
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
