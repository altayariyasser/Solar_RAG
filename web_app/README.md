# Solar IQ HTML dashboard

This is a separate web interface for the saved-model RAG. It does not replace
or modify `streamlit_app.py`.

## Run

From the project root:

```powershell
.\.venv\Scripts\python.exe web_app\server.py
```

Open <http://127.0.0.1:8000>.

The browser sends each conversational question to `POST /api/analyze`. The
Python service resolves its location and date, retrieves historical or live
features, calls the saved notebook models, and returns the business summary and
decision drivers.

Optional environment variables:

- `PORT`: server port, default `8000`
- `SOLAR_WEB_HOST`: bind address, default `127.0.0.1`
- `OLLAMA_API_KEY`: enables the optional generated explanation
- `OLLAMA_HOST`: defaults to `https://ollama.com`
- `OLLAMA_MODEL`: defaults to `gpt-oss:20b`

## Container

Build from the project root so the model and data directories are included:

```powershell
docker build -f web_app/Dockerfile -t solar-iq-web .
docker run --rm -p 8000:8000 solar-iq-web
```
