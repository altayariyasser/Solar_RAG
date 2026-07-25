"""Tests for the separate Solar IQ HTML dashboard API helpers."""

import unittest

from web_app.server import official_aqi_label, public_result


class WebDashboardTests(unittest.TestCase):
    def test_official_aqi_categories_distinguish_104_and_301(self):
        self.assertEqual(
            official_aqi_label(104),
            "Unhealthy for sensitive groups",
        )
        self.assertEqual(official_aqi_label(301), "Hazardous")
        self.assertEqual(official_aqi_label(331), "Hazardous")

    def test_public_result_uses_numeric_aqi_for_display_label(self):
        response = public_result(
            {
                "status": "success",
                "predictions": {
                    "aqi_value": 240,
                    "aqi_risk_level": "Unhealthy",
                },
                "llm_response": "The AQI model estimates 240 (Unhealthy).",
                "data": {},
            }
        )

        self.assertEqual(
            response["predictions"]["aqi_risk_level"],
            "Very unhealthy",
        )
        self.assertIn("(Very unhealthy)", response["summary"])


if __name__ == "__main__":
    unittest.main()
