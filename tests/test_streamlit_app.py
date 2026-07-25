"""Streamlit rendering test for the conversational interface."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_FILE = Path(__file__).resolve().parents[1] / "streamlit_app.py"


class StreamlitConversationTests(unittest.TestCase):
    def test_market_selector_updates_the_city_profile(self):
        app = AppTest.from_file(str(APP_FILE))
        app.run(timeout=30)

        self.assertFalse(list(app.exception))
        self.assertEqual(app.selectbox[0].value, "Riyadh")

        app.selectbox[0].set_value("Jeddah").run(timeout=30)

        self.assertFalse(list(app.exception))
        headings = " ".join(str(element.value) for element in app.subheader)
        self.assertIn("Jeddah seasonal profile", headings)

    def test_chat_question_renders_explanation_and_model_results(self):
        app = AppTest.from_file(str(APP_FILE))
        app.run(timeout=30)

        self.assertFalse(list(app.exception))
        self.assertEqual(len(app.chat_input), 1)

        app.chat_input[0].set_value(
            "How much solar energy could I get in Riyadh on 2024-02-02?"
        ).run(timeout=60)

        self.assertFalse(list(app.exception))
        rendered_text = " ".join(
            str(element.value)
            for collection in (app.markdown, app.subheader)
            for element in collection
        )
        self.assertIn("Executive outlook", rendered_text)
        self.assertIn("Key results", rendered_text)
        self.assertGreaterEqual(len(app.metric), 3)


if __name__ == "__main__":
    unittest.main()
