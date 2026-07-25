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
        self.assertIn("Include a date", result["error"])

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
