"""Tests for the lightweight RAG variant backed by persisted notebook models."""

import unittest

from rag_saved_models import MODEL_BUNDLE, SavedModelSolarRAG, SavedNotebookPredictor


class SavedNotebookModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = SavedModelSolarRAG()
        cls.rag.setup()

    def test_model_bundle_loads_without_training(self):
        self.assertTrue(MODEL_BUNDLE.exists())
        predictor = SavedNotebookPredictor()
        self.assertEqual(predictor.bundle["source_notebook"], "Project_Solar.ipynb")

    def test_saved_models_work_inside_the_existing_rag_flow(self):
        result = self.rag.process_query(
            "How was the solar potential in Riyadh on February 2, 2024?"
        )

        self.assertEqual(result["status"], "success")
        self.assertGreater(result["predictions"]["solar_output_kwh"], 0)
        self.assertEqual(
            result["predictions"]["model_source"],
            "Project_Solar.ipynb saved models",
        )
        self.assertEqual(result["predictions"]["aqi_horizon"], "next day")


if __name__ == "__main__":
    unittest.main()
