# Solar Project RAG

A business-focused Streamlit assistant for solar-generation and air-quality
decisions across Riyadh, Jeddah, Mecca, Medina, and Dammam.

The assistant accepts conversational dates such as `February 2, 2026`,
`2 February 2026`, `tomorrow`, and `last Friday`. It uses the project dataset
when an exact observation is available and retrieves date/location features
from Open-Meteo for historical or near-term dates outside that dataset before
running the trained solar and AQI models.

## Run locally

Use Python 3.12 so the local environment matches Streamlit Community Cloud.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or run the container:

```bash
docker compose up --build
```

Then open `http://localhost:8501`.

## Saved notebook model RAG

The original `rag.py` remains available. A separate, shorter version loads
persisted models reproduced from `Project_Solar.ipynb`:

```bash
# Run once whenever the notebook model definitions or dataset change
python export_notebook_models.py

# Run the lightweight saved-model RAG
python rag_saved_models.py
```

The generated bundle is stored at
`models/project_solar_models.joblib`. Application startup only loads this
bundle; it does not retrain the models.

## HTML business dashboard

A separate responsive HTML/CSS/JavaScript dashboard is available in
`web_app/`. It calls the saved notebook models through a small Python API and
does not replace the Streamlit app.

```bash
python -m web_app.server
```

Then open `http://127.0.0.1:8000`. See `web_app/README.md` for the optional
environment variables and API details.

## Deploy on Streamlit Community Cloud

1. Push this entire folder to GitHub, including `data/solar_dataset.csv` and
   `data/solar_dataset.csv.gz`. The compressed copy is the deployment fallback.
2. Create or edit the Streamlit app with:
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Python: `3.12`
3. Deploy. If an older failed app was created with another Python version,
   delete it and redeploy because Community Cloud cannot change Python in place.

The root `requirements.txt` is the only Python dependency file. No
`packages.txt`, local Ollama server, PyTorch, or model download is required.

## Project layout

```text
.
├── .streamlit/config.toml
├── data/solar_dataset.csv
├── docs/                       # Previous deployment notes, kept for reference
├── rag.py                      # Retrieval and compact prediction pipeline
├── streamlit_app.py            # Community Cloud entrypoint
├── rag_app.py                  # Backward-compatible entrypoint
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Why the previous deployment failed or stayed slow

- `requirements.txt` did not include every package imported by `rag.py`.
- A second, unused dependency file contained large PyTorch and transformer
  packages, while the cloud only reads the recognized dependency file.
- `rag.py` used hard-coded `C:\Users\...` paths that do not exist on Linux.
- The CSV was ignored by Git, so a clean cloud checkout could miss its data.
- Startup downloaded a Sentence Transformer and trained models before the UI
  could render.
- `streamlit_app.py` only created a Python list; it did not run the app.

The current version uses repository-relative paths, lightweight TF-IDF
retrieval, lazy model training, and one authoritative dependency file.
