"""Small web server for the Solar IQ HTML dashboard.

The existing Streamlit application and RAG files are intentionally left
unchanged. This server exposes the saved-model RAG through a JSON endpoint and
serves the dashboard's static HTML, CSS, and JavaScript.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag_saved_models import SavedModelSolarRAG  # noqa: E402


RAG: Optional[SavedModelSolarRAG] = None
RAG_LOCK = threading.RLock()
ALLOWED_ORIGIN = os.getenv(
    "SOLAR_ALLOWED_ORIGIN",
    "https://altayariyasser.github.io",
).rstrip("/")


def get_rag() -> SavedModelSolarRAG:
    """Load the dataset and saved notebook models once."""
    global RAG
    if RAG is None:
        with RAG_LOCK:
            if RAG is None:
                system = SavedModelSolarRAG(
                    ollama_api_key=os.getenv("OLLAMA_API_KEY"),
                    ollama_host=os.getenv("OLLAMA_HOST", "https://ollama.com"),
                    ollama_model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
                )
                system.setup()
                RAG = system
    return RAG


def official_aqi_label(value: Any) -> str:
    """Map a US AQI value to the six official display categories."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if number <= 50:
        return "Good"
    if number <= 100:
        return "Moderate"
    if number <= 150:
        return "Unhealthy for sensitive groups"
    if number <= 200:
        return "Unhealthy"
    if number <= 300:
        return "Very unhealthy"
    return "Hazardous"


def portfolio_position(
    solar_value: Any,
    system: Optional[SavedModelSolarRAG] = None,
) -> Dict[str, Any]:
    """Benchmark one estimate against the project's modeled portfolio."""
    try:
        value = float(solar_value)
    except (TypeError, ValueError):
        return {
            "label": "Model estimate",
            "detail": "Portfolio benchmark unavailable",
            "percentile": None,
        }

    if system is None or system.data_loader.df_merged is None:
        return {
            "label": "Model estimate",
            "detail": "Saved notebook model",
            "percentile": None,
        }

    target = "Estimated Daily Output (kWh)"
    frame = system.data_loader.df_merged
    if target not in frame:
        return {
            "label": "Model estimate",
            "detail": "Saved notebook model",
            "percentile": None,
        }

    values = frame[target].dropna()
    percentile = int(round((values <= value).mean() * 100))
    if percentile >= 75:
        label = "High potential"
    elif percentile >= 40:
        label = "Balanced"
    else:
        label = "Lower potential"
    return {
        "label": label,
        "detail": f"{percentile}th percentile in the modeled portfolio",
        "percentile": percentile,
    }


def public_result(
    result: Dict[str, Any],
    system: Optional[SavedModelSolarRAG] = None,
) -> Dict[str, Any]:
    """Keep the useful analytical fields and normalize the AQI presentation."""
    predictions = dict(result.get("predictions") or {})
    original_risk = predictions.get("aqi_risk_level")
    if predictions.get("aqi_value") is not None:
        predictions["aqi_risk_level"] = official_aqi_label(
            predictions["aqi_value"]
        )

    summary = result.get("llm_response")
    normalized_risk = predictions.get("aqi_risk_level")
    if (
        isinstance(summary, str)
        and original_risk
        and normalized_risk
        and original_risk != normalized_risk
    ):
        summary = summary.replace(
            f"({original_risk})",
            f"({normalized_risk})",
        )

    return {
        "status": result.get("status"),
        "error": result.get("error"),
        "query": result.get("query"),
        "city": result.get("city"),
        "date": result.get("date"),
        "intents": result.get("intents") or [],
        "source_kind": result.get("source_kind"),
        "source_label": result.get("source_label"),
        "air_quality_available": result.get("air_quality_available", True),
        "summary": summary,
        "predictions": predictions,
        "business": portfolio_position(
            predictions.get("solar_output_kwh"),
            system,
        ),
        "data": result.get("data") or {},
        "interpretations": result.get("interpretations") or [],
    }


class SolarIQHandler(SimpleHTTPRequestHandler):
    """Serve static files and the RAG analysis endpoint."""

    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        """Allow the public GitHub Pages dashboard to call this API."""
        origin = self.headers.get("Origin", "").rstrip("/")
        if ALLOWED_ORIGIN == "*" or origin == ALLOWED_ORIGIN:
            self.send_header(
                "Access-Control-Allow-Origin",
                "*" if ALLOWED_ORIGIN == "*" else ALLOWED_ORIGIN,
            )
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        if urlparse(self.path).path == "/api/analyze":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            try:
                rag = get_rag()
                self._send_json(
                    {
                        "status": "ok",
                        "markets": rag.data_loader.cities,
                        "model": "Project_Solar.ipynb saved models",
                    }
                )
            except Exception as exc:
                self._send_json(
                    {"status": "error", "error": str(exc)},
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/analyze":
            self._send_json(
                {"status": "error", "error": "Endpoint not found."},
                HTTPStatus.NOT_FOUND,
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 50_000:
                raise ValueError("The request is empty or too large.")
            payload = json.loads(self.rfile.read(content_length))
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("Please enter a question.")

            context = payload.get("context")
            if not isinstance(context, dict):
                context = {}

            with RAG_LOCK:
                system = get_rag()
                result = system.process_query(question, context=context)
            response = public_result(result, system)
            status = (
                HTTPStatus.OK
                if result.get("status") == "success"
                else HTTPStatus.UNPROCESSABLE_ENTITY
            )
            self._send_json(response, status)
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json(
                {"status": "error", "error": str(exc)},
                HTTPStatus.BAD_REQUEST,
            )
        except Exception:
            self._send_json(
                {
                    "status": "error",
                    "error": (
                        "The analysis service could not complete this request. "
                        "Please try again."
                    ),
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args) -> None:
        print(f"[Solar IQ] {self.address_string()} - {format % args}")


def main() -> None:
    host = os.getenv("SOLAR_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    get_rag()
    server = ThreadingHTTPServer((host, port), SolarIQHandler)
    print(f"Solar IQ web dashboard: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSolar IQ stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
