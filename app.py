"""Hospital-grade Flask web application for AnemiaVision AI / VYRA.

Canonical unified entrypoint routing to person3_app.app while preserving
backward-compatible module exports for gunicorn (app:app), Procfile,
Dockerfile, entrypoint.sh, and test harnesses.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure person3_app and root directory are in sys.path
_REPO_ROOT = Path(__file__).resolve().parent
_PERSON3_DIR = _REPO_ROOT / "person3_app"

for _p in (str(_PERSON3_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the canonical application from person3_app
import person3_app.app as _p3_app  # noqa: E402
from person3_app.app import *  # noqa: F401, F403, E402

# Aliases for legacy compatibility
ScanRecord = getattr(_p3_app, "Scan", None)
DEFAULT_CLINICAL_ACCURACY = 92.4

# Ensure app module refers directly to canonical person3_app.app
sys.modules[__name__] = _p3_app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
