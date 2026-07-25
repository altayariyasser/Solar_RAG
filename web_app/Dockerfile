FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SOLAR_WEB_HOST=0.0.0.0 \
    SOLAR_ALLOWED_ORIGIN=https://altayariyasser.github.io \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag.py rag_saved_models.py ./
COPY data ./data
COPY models ./models
COPY web_app ./web_app

EXPOSE 8000

CMD ["python", "-m", "web_app.server"]
