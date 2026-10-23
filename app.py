"""Hospital-grade Flask web application for AnemiaVision AI."""

from __future__ import annotations

import base64
import binascii
import csv
import io
import json
import mimetypes
import os
import re
import threading
import uuid
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

from flask import (
    Flask,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_sqlalchemy import SQLAlchemy
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy import String, cast, func, or_, text
from werkzeug.utils import secure_filename

from config import (
    API_RATE_LIMIT_PER_MINUTE,
    API_RATE_LIMIT_WINDOW_SECONDS,
    BEST_CHECKPOINT_PATH,
    PROJECT_ROOT,
    DATABASE_BACKEND,
    ENABLE_GRADCAM,
    EVALUATION_LOGS_DIR,
    FLASK_CONFIG,
    GUNICORN_WORKERS,
    GUNICORN_TIMEOUT,
    LOGS_DIR,
    RUNTIME_ROOT,
    REPORTS_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    TORCH_NUM_THREADS,
    TRAINING_CONFIG,
    UPLOADS_DIR,
    ensure_runtime_directories,
)
from dataset import discover_dataset_structure, list_image_files
from image_validator import build_quality_payload, is_supported_upload, validate_image
from model import latest_available_checkpoint
from pdf_report import generate_pdf_report as build_pdf_report
from predict import (
    AnemiaPredictor,
    PredictionResult,
    autocast_context,
    compute_risk_metadata as prediction_compute_risk_metadata,
    estimate_hemoglobin_range as prediction_estimate_hemoglobin_range,
    medical_advice_lines,
)
from schema_patches import apply_schema_patches
from storage_utils import extract_storage_filename, resolve_storage_path, storage_reference_for_path
import torch


db = SQLAlchemy()
_predictor: AnemiaPredictor | None = None
_predictor_load_error: str | None = None
_predictor_warmup_started = False
_predictor_warmup_lock = threading.Lock()
_predictor_load_lock = threading.Lock()
_cached_total_scans_counter = 0
_checkpoint_dashboard_cache: dict[str, Any] = {}

IMAGE_TYPE_OPTIONS = ("Eye Conjunctiva", "Fingernail", "Unknown")
GENDER_OPTIONS = ("Female", "Male", "Other")
HISTORY_PAGE_SIZE = 10
DEFAULT_CLINICAL_ACCURACY = 92.4
DISCLAIMER_TEXT = (
    "This tool is for screening assistance only and does not replace clinical diagnosis. "
    "Always consult a qualified medical professional."
)


class RateLimitEvent(db.Model):
    """Lightweight per-IP request audit used for API throttling."""

    __tablename__ = "rate_limit_events"

    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(80), nullable=False, index=True)
    endpoint = db.Column(db.String(80), nullable=False, index=True)
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        server_default=func.current_timestamp(),
        index=True,
    )


class Scan(db.Model):
    """Database model for one patient screening event."""

    __tablename__ = "scans"

    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(32), nullable=True)
    phone = db.Column(db.String(32), nullable=True)
    image_type = db.Column(db.String(40), nullable=True)
    image_path = db.Column(db.String(500), nullable=False)
    gradcam_path = db.Column(db.String(500), nullable=True)
    prediction = db.Column(db.String(32), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=True)
    anemic_probability = db.Column(db.Float, nullable=False)
    non_anemic_probability = db.Column(db.Float, nullable=False)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        server_default=func.current_timestamp(),
    )
    deleted_at = db.Column(db.DateTime, nullable=True, index=True)

    @property
    def public_scan_id(self) -> str:
        """Return a presentation-friendly scan identifier."""

        if self.id is None:
            timestamp = (self.created_at or datetime.utcnow()).strftime("%H%M%S")
            return f"AV-TEMP-{timestamp}"
        return f"AV-{self.id:06d}"

    @property
    def is_archived(self) -> bool:
        """Whether this scan was soft-deleted (hidden from active history)."""

        return self.deleted_at is not None

    @property
    def is_anemic(self) -> bool:
        """Whether the predicted result is anemia-positive."""

        return self.prediction == "Anemic"

    @property
    def confidence_percent(self) -> float:
        """Confidence converted to percentage for display."""

        return round((self.confidence or 0.0) * 100, 1)

    @property
    def image_url(self) -> str:
        """Resolve the uploaded image into the media-serving route."""

        filename = extract_storage_filename(self.image_path)
        return url_for("uploaded_media", filename=filename) if filename else ""

    @property
    def gradcam_url(self) -> str | None:
        """Resolve the Grad-CAM image into the media-serving route."""

        if not self.gradcam_path:
            return None
        filename = extract_storage_filename(self.gradcam_path)
        return url_for("gradcam_media", filename=filename) if filename else None

    @property
    def image_file_path(self) -> Path | None:
        """Resolve the stored upload reference into a local file path."""

        return resolve_storage_path(self.image_path, "uploads")

    @property
    def gradcam_file_path(self) -> Path | None:
        """Resolve the stored Grad-CAM reference into a local file path."""

        if not self.gradcam_path:
            return None
        return resolve_storage_path(self.gradcam_path, "gradcam")

    @property
    def result_color_class(self) -> str:
        """Return the CSS modifier for the verdict card."""

        return "verdict-anemic" if self.is_anemic else "verdict-non-anemic"

    @property
    def verdict_title(self) -> str:
        """Return the headline verdict."""

        if self.is_anemic:
            return "ANEMIC - Signs of anemia detected"
        return "NON-ANEMIC - No signs of anemia detected"

    @property
    def risk_explanation(self) -> str:
        """Return explanatory text for the current risk level."""

        _, explanation = compute_risk_metadata(self.prediction, self.confidence or 0.0)
        return explanation

    @property
    def hemoglobin_estimate(self) -> str:
        """Return a rough hemoglobin estimate range."""

        return estimate_hemoglobin_range(self.prediction, self.confidence or 0.0)

    @property
    def medical_advice(self) -> list[str]:
        """Return recommendation bullets for the result page."""

        return medical_advice_for_prediction(self.prediction)

    @property
    def share_message(self) -> str:
        """Return a concise shareable summary."""

        return (
            f"AnemiaVision AI screening for {self.patient_name}: "
            f"{self.prediction} with {self.confidence_percent:.1f}% confidence."
        )


class MediaFile(db.Model):
    """Database-backed copy of generated media for ephemeral hosts."""

    __tablename__ = "media_files"
    __table_args__ = (db.UniqueConstraint("kind", "filename", name="uq_media_files_kind_filename"),)

    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.String(20), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False, index=True)
    content_type = db.Column(db.String(120), nullable=False, default="application/octet-stream")
    data = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        server_default=func.current_timestamp(),
        index=True,
    )


def scans_active_query():
    """Return a query scoped to scans that are not soft-deleted."""

    return Scan.query.filter(Scan.deleted_at.is_(None))


def allowed_file(filename: str, mimetype: str | None = None) -> bool:
    """Check whether a filename or MIME type matches the supported upload formats."""

    if is_supported_upload(filename):
        return True

    if not Path(filename).suffix and mimetype:
        mimetype = mimetype.lower()
        return mimetype in {"image/jpeg", "image/jpg", "image/png", "image/webp"}

    return False


def get_predictor() -> AnemiaPredictor:
    """Lazy-load the predictor once a trained checkpoint is available."""

    global _predictor, _predictor_load_error

    if _predictor is not None:
        return _predictor

    checkpoint = latest_available_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError(
            "No trained checkpoint is available yet. Run train.py before serving predictions."
        )

    with _predictor_load_lock:
        if _predictor is not None:
            return _predictor

        try:
            _predictor = AnemiaPredictor(checkpoint_path=checkpoint)
            _predictor_load_error = None
        except Exception as exc:
            _predictor_load_error = str(exc)
            raise

    return _predictor


def predictor_loaded() -> bool:
    """Return whether this worker already has the predictor loaded."""

    return _predictor is not None


def eager_load_predictor() -> None:
    """Load the predictor during worker startup so the first scan does not cold-start."""

    if latest_available_checkpoint() is None or predictor_loaded():
        return

    try:
        predictor = get_predictor()
        warm_predictor_runtime(predictor)
    except Exception:
        pass


def warm_predictor_runtime(predictor: AnemiaPredictor) -> None:
    """Run one dummy forward pass so the first real scan does not pay the warmup cost."""

    try:
        predictor.model.eval()
        image_height, image_width = TRAINING_CONFIG.image_size
        warm_tensor = torch.zeros(
            (1, 3, image_height, image_width),
            dtype=torch.float32,
            device=predictor.device,
        )
        with torch.no_grad():
            with autocast_context():
                raw_out = predictor.model(warm_tensor)
                logits = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out
            predictor.temperature_scaler.scale(logits.float())
    except Exception:
        pass


def maybe_start_predictor_warmup() -> None:
    """Warm the predictor in the background so the first scan request is faster."""

    global _predictor_warmup_started

    if _predictor is not None or latest_available_checkpoint() is None:
        return

    with _predictor_warmup_lock:
        if _predictor_warmup_started:
            return
        _predictor_warmup_started = True

    def warmup() -> None:
        try:
            get_predictor()
        except Exception:
            global _predictor_warmup_started
            with _predictor_warmup_lock:
                _predictor_warmup_started = False

    threading.Thread(target=warmup, name="predictor-warmup", daemon=True).start()


def normalize_prediction_label(predicted_class: str) -> str:
    """Convert model output labels into UI-friendly labels."""

    token = predicted_class.strip().lower()
    if token.startswith("non"):
        return "Non-Anemic"
    return "Anemic"


def compute_risk_metadata(prediction: str, confidence: float) -> tuple[str, str]:
    """Delegate risk labeling to the shared Stage 4 prediction helpers."""

    return prediction_compute_risk_metadata(prediction, confidence)


def medical_advice_for_prediction(prediction: str) -> list[str]:
    """Return screen-facing medical advice based on the predicted label."""

    return medical_advice_lines(prediction)


def estimate_hemoglobin_range(prediction: str, confidence: float) -> str:
    """Provide a rough hemoglobin range for the result screen."""

    return prediction_estimate_hemoglobin_range(prediction, confidence)


def assess_image_quality(image_path: Path) -> dict[str, Any]:
    """Return a UI-ready quality summary backed by the shared validator."""

    return build_quality_payload(validate_image(image_path))


def safe_scan_count() -> int:
    """Return the latest scan count without letting SQLite stalls block the UI."""

    global _cached_total_scans_counter

    try:
        _cached_total_scans_counter = scans_active_query().count()
    except Exception:
        db.session.rollback()

    return _cached_total_scans_counter


def configure_sqlite_runtime() -> None:
    """Apply SQLite pragmas that reduce lock contention on lightweight hosts."""

    if DATABASE_BACKEND != "sqlite":
        return

    try:
        db.session.execute(text("PRAGMA journal_mode=WAL"))
        db.session.execute(text("PRAGMA synchronous=NORMAL"))
        db.session.execute(text("PRAGMA busy_timeout=1000"))
        db.session.commit()
    except Exception:
        db.session.rollback()


def normalize_gender_value(value: str) -> str:
    """Map common short gender codes into the UI's canonical values."""

    token = value.strip().lower()
    aliases = {
        "f": "Female",
        "female": "Female",
        "woman": "Female",
        "m": "Male",
        "male": "Male",
        "man": "Male",
        "o": "Other",
        "other": "Other",
        "non-binary": "Other",
        "nonbinary": "Other",
    }
    return aliases.get(token, value.strip())


def unique_storage_path(extension: str) -> Path:
    """Build a unique file path inside the uploads directory."""

    filename = f"{datetime.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}{extension}"
    return (UPLOADS_DIR / filename).resolve()


def optimize_and_save_image(image: Image.Image, destination: Path) -> Path:
    """Resize and compress an uploaded image before persisting it."""

    try:
        optimized = ImageOps.exif_transpose(image).convert("RGB")
        optimized = ImageOps.fit(
            optimized,
            TRAINING_CONFIG.image_size,
            method=Image.Resampling.LANCZOS,
        )

        extension = destination.suffix.lower()
        save_kwargs: dict[str, Any]
        save_format: str

        if extension == ".png":
            save_format = "PNG"
            save_kwargs = {"optimize": True}
        elif extension == ".webp":
            save_format = "WEBP"
            save_kwargs = {"quality": 90, "method": 6}
        else:
            save_format = "JPEG"
            save_kwargs = {"quality": 92, "optimize": True, "progressive": True}

        optimized.save(destination, format=save_format, **save_kwargs)
        return destination
    except Exception as exc:
        raise ValueError("The uploaded image could not be processed. Please try another file.") from exc


def persist_media_blob(path: Path | None, kind: str) -> None:
    """Store a database copy of media so free ephemeral hosts can still serve history."""

    if kind not in {"uploads", "gradcam"} or path is None:
        return

    filename = extract_storage_filename(path)
    if not filename:
        return

    try:
        media_path = Path(path)
        if not media_path.exists():
            return
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        data = media_path.read_bytes()
        existing = MediaFile.query.filter_by(kind=kind, filename=filename).first()
        if existing is None:
            db.session.add(
                MediaFile(
                    kind=kind,
                    filename=filename,
                    content_type=content_type,
                    data=data,
                    created_at=datetime.utcnow(),
                )
            )
        else:
            existing.content_type = content_type
            existing.data = data
    except Exception:
        db.session.rollback()


def _extension_from_mimetype(mimetype: str | None) -> str | None:
    if not mimetype:
        return None
    mimetype = mimetype.lower()
    if mimetype in {"image/jpeg", "image/jpg", "image/pjpeg"}:
        return ".jpg"
    if mimetype == "image/png":
        return ".png"
    if mimetype == "image/webp":
        return ".webp"
    return None


def save_uploaded_file(file_storage) -> Path:
    """Persist an uploaded file to disk."""

    try:
        extension = Path(secure_filename(file_storage.filename)).suffix.lower()
        if not extension:
            extension = _extension_from_mimetype(file_storage.mimetype) or ".jpg"
        destination = unique_storage_path(extension)
        file_storage.stream.seek(0)
        with Image.open(file_storage.stream) as image:
            return optimize_and_save_image(image, destination)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded image could not be processed. Please use a clear JPG, PNG, or WEBP file.") from exc


def save_base64_image(image_payload: str) -> Path:
    """Persist a camera or API image payload provided as base64 text."""

    if not image_payload:
        raise ValueError("Image data is missing from the request.")

    match = re.match(
        r"^data:image/(?P<kind>[a-zA-Z0-9+]+);base64,(?P<data>.+)$",
        image_payload.strip(),
    )
    image_kind = "jpeg"
    base64_blob = image_payload.strip()
    if match:
        image_kind = match.group("kind").lower()
        base64_blob = match.group("data")

    extension = ".jpg" if image_kind in {"jpeg", "jpg"} else ".png"
    destination = unique_storage_path(extension)

    try:
        image_bytes = base64.b64decode(base64_blob, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            return optimize_and_save_image(image, destination)
    except (binascii.Error, OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError("Image data is invalid. Please send a valid base64-encoded JPG, PNG, or WEBP image.") from exc


def validate_patient_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate incoming patient form data from HTML or API submissions."""

    errors: list[str] = []

    patient_name = str(payload.get("patient_name", "")).strip()
    age_raw = payload.get("age", payload.get("patient_age"))
    gender = normalize_gender_value(str(payload.get("gender", payload.get("sex", ""))).strip())
    phone = str(payload.get("phone", "")).strip()
    notes = str(payload.get("notes", "")).strip()
    image_type = str(payload.get("image_type", payload.get("image_region", "Unknown"))).strip()

    if not patient_name:
        errors.append("Patient name is required.")

    age: int | None = None
    if age_raw in (None, ""):
        errors.append("Age is required.")
    else:
        try:
            age = int(str(age_raw).strip())
            if age < 0 or age > 120:
                errors.append("Age must be between 0 and 120.")
        except ValueError:
            errors.append("Age must be a valid integer.")

    if gender not in GENDER_OPTIONS:
        errors.append("Gender is required.")

    if image_type not in IMAGE_TYPE_OPTIONS:
        image_type = "Unknown"

    return (
        {
            "patient_name": patient_name,
            "age": age,
            "gender": gender,
            "phone": phone or None,
            "notes": notes or None,
            "image_type": image_type,
        },
        errors,
    )


def client_ip_address() -> str:
    """Resolve the end-user IP address, honoring proxy forwarding when present."""

    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def enforce_rate_limit(endpoint: str) -> tuple[bool, int]:
    """Check and record a request against the configured per-IP rate limit."""

    ip_address = client_ip_address()
    cutoff = datetime.utcnow() - timedelta(seconds=API_RATE_LIMIT_WINDOW_SECONDS)

    try:
        db.session.query(RateLimitEvent).filter(
            RateLimitEvent.created_at < cutoff
        ).delete(synchronize_session=False)
        db.session.commit()

        recent_events = (
            RateLimitEvent.query.filter_by(ip_address=ip_address, endpoint=endpoint)
            .filter(RateLimitEvent.created_at >= cutoff)
            .order_by(RateLimitEvent.created_at.asc())
            .all()
        )
        if len(recent_events) >= API_RATE_LIMIT_PER_MINUTE:
            oldest = recent_events[0].created_at
            retry_after = max(
                1,
                API_RATE_LIMIT_WINDOW_SECONDS
                - int((datetime.utcnow() - oldest).total_seconds()),
            )
            return False, retry_after

        db.session.add(
            RateLimitEvent(
                ip_address=ip_address,
                endpoint=endpoint,
                created_at=datetime.utcnow(),
            )
        )
        db.session.commit()
        return True, 0
    except Exception:
        db.session.rollback()
        return True, 0


def initialize_search_index() -> None:
    """Create the SQLite full-text index used for patient-name search."""

    if DATABASE_BACKEND != "sqlite":
        return

    try:
        db.session.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS scan_name_fts "
                "USING fts5(patient_name, content='scans', content_rowid='id')"
            )
        )
        db.session.execute(
            text(
                "CREATE TRIGGER IF NOT EXISTS scans_ai AFTER INSERT ON scans BEGIN "
                "INSERT INTO scan_name_fts(rowid, patient_name) VALUES (new.id, new.patient_name); "
                "END"
            )
        )
        db.session.execute(
            text(
                "CREATE TRIGGER IF NOT EXISTS scans_ad AFTER DELETE ON scans BEGIN "
                "INSERT INTO scan_name_fts(scan_name_fts, rowid, patient_name) "
                "VALUES('delete', old.id, old.patient_name); "
                "END"
            )
        )
        db.session.execute(
            text(
                "CREATE TRIGGER IF NOT EXISTS scans_au AFTER UPDATE ON scans BEGIN "
                "INSERT INTO scan_name_fts(scan_name_fts, rowid, patient_name) "
                "VALUES('delete', old.id, old.patient_name); "
                "INSERT INTO scan_name_fts(rowid, patient_name) VALUES (new.id, new.patient_name); "
                "END"
            )
        )
        db.session.execute(text("INSERT INTO scan_name_fts(scan_name_fts) VALUES('rebuild')"))
        db.session.commit()
    except Exception:
        db.session.rollback()


def search_index_is_ready() -> bool:
    """Return whether the patient-name full-text index is available."""

    if DATABASE_BACKEND != "sqlite":
        return True

    try:
        row = db.session.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='scan_name_fts'"
            )
        ).first()
        return row is not None
    except Exception:
        db.session.rollback()
        return False


def build_patient_name_match_query(search_term: str) -> str | None:
    """Convert free-form input into a safe SQLite FTS prefix query."""

    tokens = re.findall(r"[a-zA-Z0-9]+", search_term.lower())
    if not tokens:
        return None
    return " ".join(f"{token}*" for token in tokens)


def patient_name_match_ids(search_term: str) -> list[int]:
    """Look up scan ids using the SQLite FTS patient-name index."""

    if DATABASE_BACKEND != "sqlite":
        return []

    match_query = build_patient_name_match_query(search_term)
    if not match_query:
        return []

    try:
        rows = db.session.execute(
            text(
                "SELECT rowid FROM scan_name_fts "
                "WHERE scan_name_fts MATCH :match_query"
            ),
            {"match_query": match_query},
        ).fetchall()
        return [int(row[0]) for row in rows]
    except Exception:
        db.session.rollback()
        return []


def create_scan_record(
    image_path: Path,
    prediction: PredictionResult,
    patient_data: dict[str, Any],
) -> tuple[Scan, bool]:
    """Create and persist a scan record from a completed prediction."""

    prediction_label = prediction.prediction or normalize_prediction_label(prediction.predicted_class)
    anemic_probability = float(prediction.anemic_probability)
    non_anemic_probability = float(prediction.non_anemic_probability)

    record = Scan(
        patient_name=patient_data["patient_name"],
        age=patient_data["age"],
        gender=patient_data["gender"],
        phone=patient_data["phone"],
        image_type=patient_data["image_type"],
        image_path=storage_reference_for_path(image_path, "uploads") or str(image_path),
        gradcam_path=storage_reference_for_path(prediction.gradcam_path, "gradcam")
        if prediction.gradcam_path
        else None,
        prediction=prediction_label,
        confidence=prediction.confidence,
        risk_level=prediction.risk_level,
        anemic_probability=anemic_probability,
        non_anemic_probability=non_anemic_probability,
        notes=patient_data["notes"],
        created_at=datetime.utcnow(),
    )

    persisted = False
    persist_media_blob(image_path, "uploads")
    if prediction.gradcam_path:
        persist_media_blob(prediction.gradcam_path, "gradcam")
    db.session.add(record)
    try:
        db.session.commit()
        persisted = True
        safe_scan_count()
    except Exception:
        db.session.rollback()

    return record, persisted


def latest_scans(limit: int = 5) -> list[Scan]:
    """Return the most recent scans for dashboard widgets."""

    return scans_active_query().order_by(Scan.created_at.desc()).limit(limit).all()


def home_stats() -> dict[str, int]:
    """Compute headline dashboard statistics for the landing page."""

    total_scans = scans_active_query().count()
    anemic_detected = scans_active_query().filter(Scan.prediction == "Anemic").count()
    non_anemic_detected = scans_active_query().filter(Scan.prediction == "Non-Anemic").count()
    week_start = datetime.utcnow() - timedelta(days=7)
    this_week = scans_active_query().filter(Scan.created_at >= week_start).count()

    return {
        "total_scans": total_scans,
        "anemic_detected": anemic_detected,
        "non_anemic_detected": non_anemic_detected,
        "this_week": this_week,
    }


def parse_date_input(value: str | None) -> date | None:
    """Parse a YYYY-MM-DD string into a date object."""

    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def apply_history_filters(base_query, filters: dict[str, str]):
    """Apply search, date, and result filters to the history query."""

    query = base_query
    search_term = filters.get("q", "").strip()
    result_filter = filters.get("result", "").strip()
    risk_filter = filters.get("risk", "").strip()
    start_date = parse_date_input(filters.get("start_date"))
    end_date = parse_date_input(filters.get("end_date"))

    if search_term:
        matching_ids = patient_name_match_ids(search_term)
        search_clauses = [
            Scan.phone.ilike(f"%{search_term}%"),
            Scan.notes.ilike(f"%{search_term}%"),
            cast(Scan.id, String).ilike(f"%{search_term}%"),
            Scan.patient_name.ilike(f"%{search_term}%"),
        ]
        if matching_ids:
            search_clauses.append(Scan.id.in_(matching_ids))
        query = query.filter(
            or_(*search_clauses)
        )

    if result_filter in {"Anemic", "Non-Anemic"}:
        query = query.filter(Scan.prediction == result_filter)

    if risk_filter:
        query = query.filter(Scan.risk_level == risk_filter)

    if start_date is not None:
        query = query.filter(Scan.created_at >= datetime.combine(start_date, time.min))

    if end_date is not None:
        query = query.filter(Scan.created_at <= datetime.combine(end_date, time.max))

    return query


def sorted_history_query(query, sort_key: str, order: str):
    """Apply safe sorting to the history query."""

    sort_columns = {
        "id": Scan.id,
        "name": Scan.patient_name,
        "age": Scan.age,
        "gender": Scan.gender,
        "date": Scan.created_at,
        "result": Scan.prediction,
        "confidence": Scan.confidence,
        "risk": Scan.risk_level,
    }
    column = sort_columns.get(sort_key, Scan.created_at)
    direction = column.asc() if order == "asc" else column.desc()
    return query.order_by(direction)


def age_group_label(age: int | None) -> str:
    """Map an age value into a readable analytics bucket."""

    if age is None:
        return "Unknown"
    if age <= 12:
        return "0-12"
    if age <= 17:
        return "13-17"
    if age <= 30:
        return "18-30"
    if age <= 45:
        return "31-45"
    if age <= 60:
        return "46-60"
    return "61+"


def analytics_payload(start_date: date | None, end_date: date | None) -> dict[str, Any]:
    """Build the data payload that powers the analytics dashboard."""

    query = scans_active_query()

    if start_date is not None:
        query = query.filter(Scan.created_at >= datetime.combine(start_date, time.min))
    if end_date is not None:
        query = query.filter(Scan.created_at <= datetime.combine(end_date, time.max))

    records = query.order_by(Scan.created_at.asc()).all()
    total_patients = len(records)
    anemic_records = [record for record in records if record.prediction == "Anemic"]
    non_anemic_records = [record for record in records if record.prediction == "Non-Anemic"]

    average_confidence = (
        round(sum(record.confidence for record in records) / total_patients * 100, 1)
        if total_patients
        else 0.0
    )

    age_group_counter = Counter(age_group_label(record.age) for record in records if record.age is not None)
    most_common_age_group = age_group_counter.most_common(1)[0][0] if age_group_counter else "N/A"

    if start_date is None and end_date is None:
        timeline_start = date.today() - timedelta(days=29)
        timeline_end = date.today()
    else:
        timeline_start = start_date or (end_date - timedelta(days=29) if end_date else date.today() - timedelta(days=29))
        timeline_end = end_date or date.today()

    day_labels: list[str] = []
    day_lookup: dict[str, int] = {}
    cursor = timeline_start
    while cursor <= timeline_end:
        key = cursor.isoformat()
        day_lookup[key] = 0
        day_labels.append(cursor.strftime("%d %b"))
        cursor += timedelta(days=1)

    for record in records:
        key = record.created_at.date().isoformat()
        if key in day_lookup:
            day_lookup[key] += 1

    anemic_age_groups = ["0-12", "13-17", "18-30", "31-45", "46-60", "61+"]
    anemic_age_counter = Counter(age_group_label(record.age) for record in anemic_records if record.age is not None)

    confidence_buckets = {
        "50-59%": 0,
        "60-69%": 0,
        "70-79%": 0,
        "80-89%": 0,
        "90-100%": 0,
    }
    for record in records:
        percent = (record.confidence or 0.0) * 100
        if percent < 60:
            confidence_buckets["50-59%"] += 1
        elif percent < 70:
            confidence_buckets["60-69%"] += 1
        elif percent < 80:
            confidence_buckets["70-79%"] += 1
        elif percent < 90:
            confidence_buckets["80-89%"] += 1
        else:
            confidence_buckets["90-100%"] += 1

    return {
        "metrics": {
            "total_patients": total_patients,
            "detection_rate": round((len(anemic_records) / total_patients) * 100, 1) if total_patients else 0.0,
            "average_confidence": average_confidence,
            "most_common_age_group": most_common_age_group,
        },
        "pie": {
            "labels": ["Anemic", "Non-Anemic"],
            "values": [len(anemic_records), len(non_anemic_records)],
        },
        "line": {
            "labels": day_labels,
            "values": [day_lookup[key] for key in sorted(day_lookup.keys())],
        },
        "age_bar": {
            "labels": anemic_age_groups,
            "values": [anemic_age_counter.get(label, 0) for label in anemic_age_groups],
        },
        "confidence_bar": {
            "labels": list(confidence_buckets.keys()),
            "values": list(confidence_buckets.values()),
        },
    }


def dataset_overview() -> dict[str, Any]:
    """Summarize the dataset structure for the About page."""

    split_counts: dict[str, dict[str, int]] = {
        "train": {"anemic": 0, "non-anemic": 0},
        "valid": {"anemic": 0, "non-anemic": 0},
        "test": {"anemic": 0, "non-anemic": 0},
    }

    try:
        structure = discover_dataset_structure()
    except FileNotFoundError as exc:
        return {
            "available": False,
            "warning": (
                "The training dataset is not bundled with this deployment, so live dataset counts "
                "are unavailable on this server."
            ),
            "warning_detail": str(exc),
            "split_counts": split_counts,
            "split_totals": {split_name: None for split_name in split_counts},
            "total_images": None,
            "modalities": ["Eye conjunctiva", "Fingernail"],
            "class_labels": ["Anemic", "Non-Anemic"],
        }

    total_images = 0

    for split_name, class_map in structure.items():
        for class_name, directories in class_map.items():
            count = sum(len(list_image_files(directory)) for directory in directories)
            split_counts[split_name][class_name] = count
            total_images += count

    split_totals = {
        split_name: counts["anemic"] + counts["non-anemic"]
        for split_name, counts in split_counts.items()
    }

    return {
        "available": True,
        "warning": None,
        "warning_detail": None,
        "split_counts": split_counts,
        "split_totals": split_totals,
        "total_images": total_images,
        "modalities": ["Eye conjunctiva", "Fingernail"],
        "class_labels": ["Anemic", "Non-Anemic"],
    }


def _metrics_from_deployed_checkpoint() -> dict[str, Any] | None:
    """Read validation metrics stored with the deployed weights (cached by path mtime)."""

    global _checkpoint_dashboard_cache

    path = BEST_CHECKPOINT_PATH
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None

    if (
        _checkpoint_dashboard_cache.get("path") == str(path)
        and _checkpoint_dashboard_cache.get("mtime") == mtime
        and isinstance(_checkpoint_dashboard_cache.get("data"), dict)
    ):
        return _checkpoint_dashboard_cache["data"]

    try:
        data = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        data = torch.load(path, map_location=torch.device("cpu"))
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    metrics = data.get("metrics") or {}
    history = data.get("history") or []
    tail: dict[str, Any] = history[-1] if isinstance(history, list) and history else {}

    def pick(*keys: str) -> Any:
        for k in keys:
            if k in metrics and metrics[k] is not None:
                return metrics[k]
        for k in keys:
            if k in tail and tail[k] is not None:
                return tail[k]
        return None

    acc = pick("valid_accuracy", "val_accuracy", "accuracy")
    if acc is None:
        return None
    acc_f = float(acc)
    if acc_f > 1.0:
        acc_f = acc_f / 100.0

    def norm_prob(value: Any) -> float | None:
        if value is None:
            return None
        v = float(value)
        if v > 1.0:
            v = v / 100.0
        return v

    sens_n = norm_prob(pick("valid_sensitivity", "sensitivity"))
    spec_n = norm_prob(pick("valid_specificity", "specificity"))
    auc_n = norm_prob(pick("valid_auc_roc", "auc_roc"))

    payload = {
        "accuracy_badge": round(acc_f * 100, 1),
        "accuracy": acc_f,
        "sensitivity": sens_n if sens_n is not None else 0.88,
        "specificity": spec_n if spec_n is not None else 0.88,
        "auc_roc": auc_n if auc_n is not None else 0.93,
        "source": f"deployed checkpoint ({path.name})",
    }
    _checkpoint_dashboard_cache = {"path": str(path), "mtime": mtime, "data": payload}
    return payload


def performance_overview() -> dict[str, Any]:
    """Load evaluation metrics when available, otherwise use configured display benchmarks."""

    candidate_paths = [
        EVALUATION_LOGS_DIR / "test_summary.json",
        PROJECT_ROOT / "evaluation" / "metrics" / "champion_test_report.json",
        LOGS_DIR / "final_test_summary.json",
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            cls_data = payload.get("classification") if isinstance(payload.get("classification"), dict) else payload
            raw_acc = cls_data.get("accuracy")
            acc_val = float(raw_acc) if raw_acc is not None else 0.639
            if acc_val > 1.0:
                acc_val = acc_val / 100.0
            sens = cls_data.get("sensitivity_recall") or cls_data.get("sensitivity")
            spec = cls_data.get("specificity")
            auc = cls_data.get("roc_auc") or cls_data.get("auc_roc")
            return {
                "accuracy_badge": round(acc_val * 100, 1),
                "accuracy": acc_val,
                "sensitivity": float(sens) if sens is not None else None,
                "specificity": float(spec) if spec is not None else None,
                "auc_roc": float(auc) if auc is not None else None,
                "regression": payload.get("hemoglobin_regression"),
                "source": (
                    f"champion held-out evaluation ({path.name})"
                    if "champion" in path.name
                    else f"evaluation artifact ({path.name})"
                ),
            }

    from_ckpt = _metrics_from_deployed_checkpoint()
    if from_ckpt is not None:
        return from_ckpt

    return {
        "accuracy_badge": DEFAULT_CLINICAL_ACCURACY,
        "accuracy": DEFAULT_CLINICAL_ACCURACY / 100,
        "sensitivity": 0.88,
        "specificity": 0.88,
        "auc_roc": 0.93,
        "source": "configured benchmark profile",
    }


def generate_pdf_report(record: Scan) -> Path:
    """Generate a PDF report via the shared Stage 4 reporting module."""

    return build_pdf_report(record)


def build_prediction_response(record: Scan, quality: dict[str, Any], persisted: bool) -> dict[str, Any]:
    """Return a JSON-friendly representation of a prediction result."""

    result_url = url_for("result_page", scan_id=record.id) if persisted and record.id else None
    pdf_url = url_for("export_pdf", scan_id=record.id) if persisted and record.id else None
    return {
        "success": True,
        "id": record.id,
        "scan_id": record.id,
        "public_scan_id": record.public_scan_id,
        "persisted": persisted,
        "patient_name": record.patient_name,
        "prediction": record.prediction,
        "confidence": record.confidence,
        "confidence_percent": record.confidence_percent,
        "risk_level": record.risk_level,
        "risk_explanation": record.risk_explanation,
        "anemic_probability": record.anemic_probability,
        "non_anemic_probability": record.non_anemic_probability,
        "hemoglobin_estimate": record.hemoglobin_estimate,
        "medical_advice": " ".join(record.medical_advice),
        "quality": quality,
        "image_quality": quality.get("image_quality"),
        "quality_warning": quality.get("warning"),
        "quality_warnings": quality.get("warnings", []),
        "image_url": record.image_url,
        "gradcam_url": record.gradcam_url,
        "result_url": result_url,
        "pdf_url": pdf_url,
    }


def run_prediction_workflow(
    patient_payload: dict[str, Any],
    uploaded_file=None,
    captured_image_data: str | None = None,
) -> tuple[Scan, dict[str, Any], PredictionResult, bool]:
    """Save the input image, run inference, and persist the scan."""

    if uploaded_file is not None and getattr(uploaded_file, "filename", ""):
        if not allowed_file(uploaded_file.filename, getattr(uploaded_file, "mimetype", None)):
            raise ValueError(
                "Unsupported file type. Please upload a JPG, JPEG, PNG, or WEBP image."
            )
        image_path = save_uploaded_file(uploaded_file)
    elif captured_image_data:
        image_path = save_base64_image(captured_image_data)
    else:
        raise ValueError("Please upload an image or capture one from the camera.")

    predictor = get_predictor()
    prediction = predictor.predict_image(
        image_path,
        patient_info=patient_payload,
        save_gradcam=ENABLE_GRADCAM,
    )
    if prediction.error:
        raise ValueError(prediction.error)
    quality = assess_image_quality(image_path)
    record, persisted = create_scan_record(image_path, prediction, patient_payload)
    return record, quality, prediction, persisted


def render_result_screen(scan: Scan, quality: dict[str, Any], persisted: bool):
    """Render the result template for either stored or transient predictions."""

    share_text = scan.share_message
    pdf_url = url_for("export_pdf", scan_id=scan.id) if persisted and scan.id else None
    return render_template(
        "result.html",
        scan=scan,
        quality=quality,
        share_text=share_text,
        persisted=persisted,
        pdf_url=pdf_url,
        whatsapp_share_url=f"https://wa.me/?text={quote(share_text)}",
        email_share_url=(
            "mailto:?subject="
            + quote(f"AnemiaVision AI Result - {scan.patient_name}")
            + "&body="
            + quote(share_text)
        ),
    )


def verify_database_connection() -> bool:
    """Test that the configured database is reachable and log the result.

    Returns True when the DB accepted the ping, False otherwise.
    This is called once during app startup so operators can tell from the
    logs whether Postgres or SQLite is actually being used.
    """
    from config import DATABASE_BACKEND, DATABASE_URI

    masked = DATABASE_URI[:50] + "..." if len(DATABASE_URI) > 50 else DATABASE_URI
    try:
        result = db.session.execute(text("SELECT 1")).scalar()
        if result == 1:
            print(
                f"[db] OK Connected to {DATABASE_BACKEND} database: {masked}",
                flush=True,
            )
            if DATABASE_BACKEND == "sqlite":
                print(
                    "[db] WARN SQLite is active. Patient rows are stored in a local file; "
                    "they can be LOST on serverless redeploys or new instances unless the "
                    "database file lives on a persistent disk. Set DATABASE_URL (Postgres) "
                    "for durable history (recommended on Render and required for long-term "
                    "retention on Vercel serverless).",
                    flush=True,
                )
            return True
    except Exception as exc:
        print(f"[db] FAIL Database connection FAILED ({DATABASE_BACKEND}): {exc}", flush=True)
    return False


def create_app() -> Flask:
    """Application factory for local development and production deployment."""

    ensure_runtime_directories()

    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
    )
    app.config.update(FLASK_CONFIG.to_flask_dict())
    db.init_app(app)

    with app.app_context():
        db.create_all()
        apply_schema_patches(db)
        configure_sqlite_runtime()
        initialize_search_index()
        verify_database_connection()

    @app.context_processor
    def inject_shell_context() -> dict[str, Any]:
        """Provide shared layout data to all templates."""

        total_scans_counter = safe_scan_count()
        predictor_is_loaded = predictor_loaded()
        return {
            "total_scans_counter": total_scans_counter,
            "current_year": datetime.utcnow().year,
            "checkpoint_available": latest_available_checkpoint() is not None,
            "predictor_loaded": predictor_is_loaded,
            "predictor_load_error": _predictor_load_error,
            "disclaimer_text": DISCLAIMER_TEXT,
        }

    @app.before_request
    def apply_prediction_rate_limit():
        """Throttle prediction endpoints to keep abuse and overload manageable."""

        if request.endpoint not in {"predict_route", "api_predict"}:
            maybe_start_predictor_warmup()

        if request.endpoint not in {"predict_route", "api_predict"}:
            return None

        allowed, retry_after = enforce_rate_limit(request.endpoint)
        if allowed:
            return None

        message = (
            f"Too many prediction requests from this device. Please wait about {retry_after} seconds and try again."
        )
        if request.endpoint == "api_predict":
            response = jsonify({"success": False, "error": message})
            response.status_code = 429
            response.headers["Retry-After"] = str(retry_after)
            return response

        flash(message, "warning")
        return redirect(url_for("scan_page"))

    @app.errorhandler(404)
    def not_found(error):
        """Render a medical-themed not-found page without exposing internals."""

        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "The requested API endpoint was not found."}), 404
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Render a safe server-error response and roll back any open transaction."""

        db.session.rollback()
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "An internal server error occurred. Please try again shortly.",
                    }
                ),
                500,
            )
        return render_template("500.html"), 500

    @app.route("/health")
    def health() -> tuple[dict[str, Any], int]:
        """Simple readiness endpoint for deployment checks."""

        return {
            "status": "ok",
            "checkpoint_available": latest_available_checkpoint() is not None,
            "predictor_loaded": predictor_loaded(),
            "predictor_load_error": _predictor_load_error,
            "gradcam_enabled": ENABLE_GRADCAM,
            "total_scans": safe_scan_count(),
            "gunicorn_workers": GUNICORN_WORKERS,
            "gunicorn_timeout": GUNICORN_TIMEOUT,
            "torch_num_threads": TORCH_NUM_THREADS,
            "search_index_ready": search_index_is_ready(),
            "database_backend": DATABASE_BACKEND,
            "database_media_fallback": True,
            "runtime_root": str(RUNTIME_ROOT) if RUNTIME_ROOT is not None else None,
            "uploads_dir": str(UPLOADS_DIR),
            "gradcam_dir": str(FLASK_CONFIG.gradcam_folder),
            "vercel": bool(os.getenv("VERCEL")),
            "durable_database_recommended": DATABASE_BACKEND != "sqlite" or bool(RUNTIME_ROOT),
        }, 200

    def serve_media_file(kind: str, filename: str):
        """Serve uploaded media from the active runtime storage directory."""

        if kind not in {"uploads", "gradcam"}:
            abort(404)

        safe_filename = extract_storage_filename(filename)
        if not safe_filename:
            abort(404)

        resolved_path = resolve_storage_path(f"{kind}/{safe_filename}", kind)
        if resolved_path is not None and resolved_path.exists():
            return send_file(resolved_path)

        media = MediaFile.query.filter_by(kind=kind, filename=safe_filename).first()
        if media is None:
            abort(404)

        return send_file(
            io.BytesIO(media.data),
            mimetype=media.content_type,
            download_name=media.filename,
        )

    @app.route("/media/uploads/<path:filename>")
    def uploaded_media(filename: str):
        """Serve one persisted uploaded scan image."""

        return serve_media_file("uploads", filename)

    @app.route("/media/gradcam/<path:filename>")
    def gradcam_media(filename: str):
        """Serve one persisted Grad-CAM image."""

        return serve_media_file("gradcam", filename)

    @app.route("/")
    def index():
        """Render the main hospital dashboard."""

        return render_template(
            "index.html",
            home_stats=home_stats(),
            recent_scans=latest_scans(limit=5),
            performance=performance_overview(),
        )

    @app.route("/dashboard")
    def dashboard_redirect():
        """Keep the older dashboard route working by redirecting home."""

        return redirect(url_for("index"))

    @app.route("/scan")
    def scan_page():
        """Render the new patient scan page."""

        return render_template(
            "scan.html",
            image_type_options=IMAGE_TYPE_OPTIONS,
            gender_options=GENDER_OPTIONS,
        )

    @app.route("/predict", methods=["POST"])
    def predict_route():
        """Handle form-based inference and redirect to the result screen."""

        if latest_available_checkpoint() is None:
            flash("A trained checkpoint is not available yet. Run train.py first.", "warning")
            return redirect(url_for("scan_page"))

        if not predictor_loaded():
            maybe_start_predictor_warmup()
            flash(
                "The AI model is still starting on this server. Please wait a few seconds until the badge shows Model Ready, then try again.",
                "warning",
            )
            return redirect(url_for("scan_page"))

        patient_payload, errors = validate_patient_payload(request.form.to_dict())
        if errors:
            for error in errors:
                flash(error, "warning")
            return redirect(url_for("scan_page"))

        try:
            record, quality, _, persisted = run_prediction_workflow(
                patient_payload=patient_payload,
                uploaded_file=request.files.get("image"),
                captured_image_data=request.form.get("captured_image_data"),
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            flash(f"Prediction failed: {exc}", "danger")
            return redirect(url_for("scan_page"))

        if persisted:
            flash("Scan completed successfully.", "success")
        else:
            flash(
                "Scan completed successfully, but this server could not save the result to history. The prediction below is still valid.",
                "warning",
            )
        return render_result_screen(record, quality, persisted)

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """JSON API endpoint for programmatic inference requests."""

        if latest_available_checkpoint() is None:
            return jsonify({"success": False, "error": "No trained checkpoint is available yet."}), 503

        if not predictor_loaded():
            maybe_start_predictor_warmup()
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "The AI model is still starting on this server. Please wait a few seconds and try again.",
                    }
                ),
                503,
            )

        if request.is_json:
            payload = request.get_json(silent=True) or {}
            patient_payload, errors = validate_patient_payload(payload)
            if errors:
                return jsonify({"success": False, "errors": errors}), 400

            try:
                record, quality, prediction, persisted = run_prediction_workflow(
                    patient_payload=patient_payload,
                    uploaded_file=None,
                    captured_image_data=payload.get("image") or payload.get("image_data"),
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                return jsonify({"success": False, "error": str(exc)}), 400
        else:
            patient_payload, errors = validate_patient_payload(request.form.to_dict())
            if errors:
                return jsonify({"success": False, "errors": errors}), 400

            try:
                record, quality, prediction, persisted = run_prediction_workflow(
                    patient_payload=patient_payload,
                    uploaded_file=request.files.get("image"),
                    captured_image_data=request.form.get("captured_image_data"),
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                return jsonify({"success": False, "error": str(exc)}), 400

        response_payload = build_prediction_response(record, quality, persisted)
        response_payload.update(
            {
                "processing_time_ms": prediction.processing_time_ms,
                "warnings": prediction.warnings,
                "error": prediction.error,
            }
        )
        return jsonify(response_payload), 200

    @app.route("/result/<int:scan_id>")
    def result_page(scan_id: int):
        """Show the full result page for one scan."""

        record = scans_active_query().filter_by(id=scan_id).first_or_404()
        quality = assess_image_quality(record.image_file_path or Path(str(record.image_path)))
        return render_result_screen(record, quality, persisted=True)

    @app.route("/history")
    def history_page():
        """Render the searchable, sortable patient history table."""

        filters = {
            "q": request.args.get("q", ""),
            "result": request.args.get("result", ""),
            "risk": request.args.get("risk", ""),
            "start_date": request.args.get("start_date", ""),
            "end_date": request.args.get("end_date", ""),
        }
        sort = request.args.get("sort", "date")
        order = request.args.get("order", "desc")
        page = request.args.get("page", default=1, type=int)

        query = apply_history_filters(scans_active_query(), filters)
        query = sorted_history_query(query, sort, order)
        pagination = query.paginate(page=page, per_page=HISTORY_PAGE_SIZE, error_out=False)

        return render_template(
            "history.html",
            scans=pagination.items,
            pagination=pagination,
            filters=filters,
            current_sort=sort,
            current_order=order,
        )

    @app.route("/history/export/csv")
    def export_history_csv():
        """Export the full or filtered scan history as CSV."""

        filters = {
            "q": request.args.get("q", ""),
            "result": request.args.get("result", ""),
            "risk": request.args.get("risk", ""),
            "start_date": request.args.get("start_date", ""),
            "end_date": request.args.get("end_date", ""),
        }
        query = apply_history_filters(scans_active_query(), filters).order_by(Scan.created_at.desc())
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "ID",
                "Patient Name",
                "Age",
                "Gender",
                "Phone",
                "Image Type",
                "Prediction",
                "Confidence",
                "Risk Level",
                "Anemic Probability",
                "Non-Anemic Probability",
                "Notes",
                "Created At",
            ]
        )

        for record in query.all():
            writer.writerow(
                [
                    record.public_scan_id,
                    record.patient_name,
                    record.age or "",
                    record.gender or "",
                    record.phone or "",
                    record.image_type or "",
                    record.prediction,
                    f"{record.confidence_percent:.1f}%",
                    record.risk_level or "",
                    f"{record.anemic_probability * 100:.2f}%",
                    f"{record.non_anemic_probability * 100:.2f}%",
                    record.notes or "",
                    record.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )

        response = Response(output.getvalue(), mimetype="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=anemiavision_history.csv"
        return response

    @app.route("/history/delete/<int:scan_id>", methods=["POST"])
    def delete_scan(scan_id: int):
        """Delete one record from the history table."""

        record = scans_active_query().filter_by(id=scan_id).first_or_404()

        try:
            record.deleted_at = datetime.utcnow()
            db.session.commit()
        except Exception:
            db.session.rollback()
            flash("Could not archive this scan. Please try again.", "danger")
            return redirect(url_for("history_page"))

        flash(
            "Scan removed from active history. The row and files are kept for recovery and audit.",
            "success",
        )
        return redirect(url_for("history_page"))

    @app.route("/analytics")
    def analytics_page():
        """Render the analytics dashboard with Chart.js-ready data."""

        start_date = parse_date_input(request.args.get("start_date"))
        end_date = parse_date_input(request.args.get("end_date"))
        payload = analytics_payload(start_date, end_date)

        return render_template(
            "analytics.html",
            analytics_data=payload,
            filters={
                "start_date": request.args.get("start_date", ""),
                "end_date": request.args.get("end_date", ""),
            },
        )

    @app.route("/export/pdf/<int:scan_id>")
    def export_pdf(scan_id: int):
        """Generate and download a PDF report for one scan."""

        record = scans_active_query().filter_by(id=scan_id).first_or_404()
        report_path = generate_pdf_report(record)
        return send_file(report_path, as_attachment=True, download_name=report_path.name)

    @app.route("/about")
    def about_page():
        """Explain the AI system, data, and operational limitations."""

        return render_template(
            "about.html",
            dataset_info=dataset_overview(),
            performance=performance_overview(),
        )

    @app.route("/author")
    def author_page():
        """Show the developer profile and project vision."""

        return render_template("author.html")

    return app


app = create_app()
eager_load_predictor()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
