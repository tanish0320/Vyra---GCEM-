"""Gunicorn runtime settings for Render and other production hosts."""

from __future__ import annotations

import os


bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = 1
timeout = int(os.getenv("ANEMIA_GUNICORN_TIMEOUT", "180"))
graceful_timeout = int(os.getenv("ANEMIA_GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = 5
preload_app = True
