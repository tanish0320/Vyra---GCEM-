FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    ANEMIA_GUNICORN_WORKERS=1 \
    ANEMIA_GUNICORN_TIMEOUT=180

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-full.txt .
RUN pip install --upgrade pip && pip install -r requirements-full.txt

COPY . .

# Make the entrypoint executable
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"7860\")}/health', timeout=3)" || exit 1

# Use entrypoint.sh so DB migrations always run before gunicorn starts
CMD ["/app/entrypoint.sh"]
