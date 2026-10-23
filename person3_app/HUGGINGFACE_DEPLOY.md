# Hugging Face Spaces Deployment

Hugging Face Spaces is the best free deployment target for this project because
the free CPU Basic runtime has enough memory for PyTorch inference.

## Create the Space

1. Go to Hugging Face and create a new Space.
2. Choose **Docker** as the Space SDK.
3. Upload or push this repository to the Space.
4. In the Space README metadata, use:

```yaml
---
title: AnemiaVision AI
sdk: docker
app_port: 7860
---
```

## Environment Variables

For saved patient history, set a managed Postgres database URL:

```text
DATABASE_URL=<your Supabase/Neon/Postgres connection string>
FLASK_SECRET_KEY=<long random secret>
ANEMIA_ENABLE_GRADCAM=true
ANEMIA_GUNICORN_WORKERS=1
ANEMIA_GUNICORN_TIMEOUT=180
ANEMIA_TORCH_THREADS=1
```

Without `DATABASE_URL`, the app falls back to SQLite inside the container and
history can disappear when the Space restarts.

## Important Free-Tier Limitation

Free Hugging Face Spaces can sleep when unused. When it wakes, the first request
may take time because PyTorch and the model checkpoint need to load again.
For a website that never sleeps, upgrade the Space hardware or use a small paid
VM/VPS with at least 2 GB RAM.
