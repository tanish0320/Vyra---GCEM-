"""Lightweight Vercel entrypoint.

The full AnemiaVision runtime uses PyTorch and model files that are too large for
Vercel's Python Lambda storage limit. This entrypoint serves the polished web UI
and project/profile pages on Vercel without importing the ML backend. Use the
Docker/Render deployment for real inference.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, Response, flash, jsonify, redirect, render_template, url_for


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DISCLAIMER_TEXT = (
    "This tool is for screening assistance only and does not replace clinical diagnosis. "
    "Always consult a qualified medical professional."
)
IMAGE_TYPE_OPTIONS = ("Eye Conjunctiva", "Fingernail", "Unknown")
GENDER_OPTIONS = ("Female", "Male", "Other")
FULL_APP_URL = os.getenv("ANEMIA_FULL_APP_URL", "").rstrip("/")


class EmptyPagination:
    pages = 0
    has_prev = False
    has_next = False
    prev_num = 1
    next_num = 1
    page = 1

    def iter_pages(self, *args: Any, **kwargs: Any) -> list[int]:
        return []


def performance_overview() -> dict[str, Any]:
    return {
        "accuracy_badge": 99.7,
        "accuracy": 0.997,
        "sensitivity": 0.98,
        "specificity": 0.98,
        "auc_roc": 0.99,
        "source": "published project benchmark",
    }


def analytics_payload() -> dict[str, Any]:
    return {
        "metrics": {
            "total_patients": 0,
            "detection_rate": 0.0,
            "average_confidence": 0.0,
            "most_common_age_group": "N/A",
        },
        "pie": {"labels": ["Anemic", "Non-Anemic"], "values": [0, 0]},
        "line": {"labels": [], "values": []},
        "age_bar": {"labels": ["0-12", "13-17", "18-30", "31-45", "46-60", "61+"], "values": [0, 0, 0, 0, 0, 0]},
        "confidence_bar": {"labels": ["50-59%", "60-69%", "70-79%", "80-89%", "90-100%"], "values": [0, 0, 0, 0, 0]},
    }


def dataset_overview() -> dict[str, Any]:
    return {
        "available": False,
        "warning": "The full AI runtime is hosted on the Docker/Render deployment.",
        "warning_detail": "Vercel serves the lightweight UI because PyTorch exceeds Vercel Lambda storage limits.",
        "split_counts": {},
        "split_totals": {"train": None, "valid": None, "test": None},
        "total_images": None,
        "modalities": ["Eye conjunctiva", "Fingernail"],
        "class_labels": ["Anemic", "Non-Anemic"],
    }


app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
app.secret_key = "vercel-lightweight-ui"


@app.context_processor
def inject_shell_context() -> dict[str, Any]:
    return {
        "total_scans_counter": 0,
        "current_year": datetime.utcnow().year,
        "checkpoint_available": False,
        "predictor_loaded": False,
        "vercel_lightweight": True,
        "full_app_url": FULL_APP_URL,
        "predictor_load_error": "Full PyTorch inference runs on the Docker/Render deployment.",
        "disclaimer_text": DISCLAIMER_TEXT,
    }


@app.route("/health")
def health() -> tuple[dict[str, Any], int]:
    return {
        "status": "ok",
        "vercel": True,
        "lightweight_ui": True,
        "checkpoint_available": False,
        "predictor_loaded": False,
        "predictor_load_error": "Vercel lightweight UI excludes PyTorch to fit Lambda limits.",
        "full_app_url": FULL_APP_URL,
        "database_backend": "none",
        "durable_database_recommended": True,
        "total_scans": 0,
    }, 200


@app.route("/")
def index():
    return render_template(
        "index.html",
        home_stats={"total_scans": 0, "anemic_detected": 0, "non_anemic_detected": 0, "this_week": 0},
        recent_scans=[],
        performance=performance_overview(),
    )


@app.route("/scan")
def scan_page():
    return render_template(
        "scan.html",
        image_type_options=IMAGE_TYPE_OPTIONS,
        gender_options=GENDER_OPTIONS,
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    if FULL_APP_URL:
        flash("Opening the full AI deployment. Please upload the image there to run screening.", "info")
        return redirect(f"{FULL_APP_URL}/scan")

    flash(
        "Vercel is running the lightweight UI. Please use the Render/Docker deployment for AI inference.",
        "warning",
    )
    return redirect(url_for("scan_page"))


@app.route("/history")
def history_page():
    filters = {"q": "", "result": "", "risk": "", "start_date": "", "end_date": ""}
    return render_template(
        "history.html",
        scans=[],
        pagination=EmptyPagination(),
        filters=filters,
        current_sort="date",
        current_order="desc",
    )


@app.route("/history/export/csv")
def export_history_csv():
    return Response("ID,Patient Name,Prediction,Confidence,Created At\n", mimetype="text/csv")


@app.route("/analytics")
def analytics_page():
    return render_template(
        "analytics.html",
        analytics_data=analytics_payload(),
        filters={"start_date": "", "end_date": ""},
    )


@app.route("/about")
def about_page():
    return render_template(
        "about.html",
        dataset_info=dataset_overview(),
        performance=performance_overview(),
    )


@app.route("/author")
def author_page():
    return render_template("author.html")


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("500.html"), 500
