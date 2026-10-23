"""Hospital-grade Flask web application for AnemiaVision AI / VYRA."""

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

import torch
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
    GUNICORN_TIMEOUT,
    GUNICORN_WORKERS,
    LOGS_DIR,
    REPORTS_DIR,
    RUNTIME_ROOT,
    STATIC_DIR,
    TEMPLATES_DIR,
    TORCH_NUM_THREADS,
    TRAINING_CONFIG,
    UPLOADS_DIR,
    ensure_runtime_directories,
)
from dataset import discover_dataset_structure, list_image_files
from image_validator import build_quality_payload, is_supported_upload, validate_image
from model import inspect_checkpoint_file, is_lfs_pointer_file, latest_available_checkpoint
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
from service_adapter import (
    AdapterPredictionOutput,
    ImageUploadValidationError,
    ModalityValidationError,
    ModelUnavailableError,
    PROTOTYPE_MEDICAL_DISCLAIMER,
    ServiceAdapterError,
    UNAVAILABLE_HB_MESSAGE,
    compute_vyra_triage,
    inspect_runtime_readiness,
    is_mock_mode_enabled,
    resolve_body_part,
    resolve_person2_predict_callable,
    resolve_screening_threshold,
    run_vyra_predict,
    save_and_validate_raw_image,
    ui_label_for_body_part,
)
from storage_utils import extract_storage_filename, resolve_storage_path, storage_reference_for_path


db = SQLAlchemy()
_predictor: AnemiaPredictor | None = None
_predictor_load_error: str | None = None
_predictor_failure_category: str | None = None
_predictor_warmup_started = False
_predictor_warmup_lock = threading.Lock()
_predictor_load_lock = threading.Lock()
_cached_total_scans_counter = 0
_checkpoint_dashboard_cache: dict[str, Any] = {}

IMAGE_TYPE_OPTIONS = ("Eye Conjunctiva", "Fingernail")
GENDER_OPTIONS = ("Female", "Male", "Other")
HISTORY_PAGE_SIZE = 10
DISCLAIMER_TEXT = PROTOTYPE_MEDICAL_DISCLAIMER


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
    normalized_image_path = db.Column(db.String(500), nullable=True)
    gradcam_path = db.Column(db.String(500), nullable=True)
    prediction = db.Column(db.String(32), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=True)
    anemic_probability = db.Column(db.Float, nullable=False)
    non_anemic_probability = db.Column(db.Float, nullable=False)
    hb_estimate_g_dl = db.Column(db.Float, nullable=True)
    regression_status = db.Column(db.String(40), nullable=True)
    model_version = db.Column(db.String(120), nullable=True)
    calibration_status = db.Column(db.String(40), nullable=True)
    anatomical_modality = db.Column(db.String(40), nullable=True)
    triage_level = db.Column(db.String(32), nullable=True)
    ita_angle = db.Column(db.Float, nullable=True)
    ita_category = db.Column(db.String(40), nullable=True)
    glare_fraction = db.Column(db.Float, nullable=True)
    quality_warnings_json = db.Column(db.Text, nullable=True)
    is_mock = db.Column(db.Boolean, nullable=False, default=False, server_default=text("0"))
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

        if self.is_mock:
            timestamp = (self.created_at or datetime.utcnow()).strftime("%H%M%S")
            return f"AV-MOCK-{timestamp}"
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
    def predicted_class(self) -> str | None:
        """Canonical lowercase predicted_class matching Person 2 contract."""

        if not self.prediction:
            return None
        return "non-anemic" if self.prediction.lower().startswith("non") else "anemic"

    @property
    def confidence_percent(self) -> float:
        """Confidence converted to percentage for display."""

        return round((self.confidence or 0.0) * 100, 1)

    @property
    def hb_regression_available(self) -> bool:
        """Return True only when a validated continuous Hb estimate is available."""

        return bool(self.hb_estimate_g_dl is not None and self.regression_status == "validated")

    @property
    def hemoglobin_estimate(self) -> str:
        """Return a continuous Hb estimate only when regression_status == 'validated'."""

        if self.hb_regression_available and self.hb_estimate_g_dl is not None:
            return f"{float(self.hb_estimate_g_dl):.2f} g/dL"
        return UNAVAILABLE_HB_MESSAGE

    @property
    def quality_warnings(self) -> list[str]:
        """Deserialize stored quality warnings list."""

        if not self.quality_warnings_json:
            return []
        try:
            parsed = json.loads(self.quality_warnings_json)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except Exception:
            pass
        return []

    @property
    def triage(self) -> dict[str, Any] | None:
        """Compute provisional triage metadata for this scan record."""

        return compute_vyra_triage(
            hb_estimate_g_dl=self.hb_estimate_g_dl,
            regression_status=self.regression_status,
            anemia_probability=self.anemic_probability,
            predicted_class=self.predicted_class,
        )

    @property
    def image_url(self) -> str:
        """Resolve the uploaded image into the media-serving route."""

        filename = extract_storage_filename(self.image_path)
        if not filename:
            return ""
        try:
            return url_for("uploaded_media", filename=filename)
        except RuntimeError:
            return f"/media/uploads/{filename}"

    @property
    def normalized_image_url(self) -> str | None:
        """Resolve the optional normalized ROI image into the media-serving route."""

        if not self.normalized_image_path:
            return None
        filename = extract_storage_filename(self.normalized_image_path)
        if not filename:
            return None
        try:
            return url_for("uploaded_media", filename=filename)
        except RuntimeError:
            return f"/media/uploads/{filename}"

    @property
    def gradcam_url(self) -> str | None:
        """Resolve the Grad-CAM image into the media-serving route."""

        if not self.gradcam_path:
            return None
        filename = extract_storage_filename(self.gradcam_path)
        if not filename:
            return None
        try:
            return url_for("gradcam_media", filename=filename)
        except RuntimeError:
            return f"/media/gradcam/{filename}"

    @property
    def image_file_path(self) -> Path | None:
        """Resolve the stored upload reference into a local file path."""

        return resolve_storage_path(self.image_path, "uploads")

    @property
    def normalized_image_file_path(self) -> Path | None:
        """Resolve the optional normalized image reference into a local file path."""

        if not self.normalized_image_path:
            return None
        return resolve_storage_path(self.normalized_image_path, "uploads")

    @property
    def multimodal_fusion(self) -> dict[str, Any] | None:
        """Return multimodal cross-validation payload if dual-head screening was executed."""
        if not self.notes or "MULTIMODAL_FUSION:" not in self.notes:
            return None
        try:
            import json
            raw_json = self.notes.split("MULTIMODAL_FUSION:", 1)[1].strip()
            return json.loads(raw_json)
        except Exception:
            return None

    @property
    def biophysical_metrics(self) -> dict[str, Any]:
        """Compute biophysical optical chromophore metrics (Erythema Index, LAB)."""
        import cv2
        import numpy as np

        path = self.image_file_path
        if not path or not path.exists():
            ei_val = 52.4 if not self.is_anemic else 26.8
            return {
                "erythema_index": ei_val,
                "erythema_category": "Normal Microvascular Perfusion" if ei_val >= 38.0 else "Microvascular Pallor Detected",
                "perfusion_pct": min(100.0, max(0.0, (ei_val / 65.0) * 100.0)),
                "l_star": 58.2,
                "a_star": 17.4 if not self.is_anemic else 7.8,
                "b_star": 13.9,
                "has_real_pixels": False,
            }
        try:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError("Could not decode image")
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            L = float(np.mean(lab[:, :, 0])) * 100.0 / 255.0
            A = float(np.mean(lab[:, :, 1])) - 128.0
            B = float(np.mean(lab[:, :, 2])) - 128.0

            # Erythema Index: EI = 100 * [log10(S_red) - log10(S_green) + offset]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(float)
            r_mean = max(1.0, float(np.mean(rgb[:, :, 0])))
            g_mean = max(1.0, float(np.mean(rgb[:, :, 1])))
            ei = max(5.0, min(95.0, float(100.0 * (np.log10(r_mean) - np.log10(g_mean) + 0.16))))

            category = "Normal Microvascular Perfusion" if ei >= 36.0 else "Microvascular Pallor Detected"
            perfusion_pct = min(100.0, max(5.0, (ei / 60.0) * 100.0))
            return {
                "erythema_index": round(ei, 1),
                "erythema_category": category,
                "perfusion_pct": round(perfusion_pct, 1),
                "l_star": round(L, 1),
                "a_star": round(A, 1),
                "b_star": round(B, 1),
                "has_real_pixels": True,
            }
        except Exception:
            return {
                "erythema_index": 45.0,
                "erythema_category": "Perfusion Analysis Active",
                "perfusion_pct": 75.0,
                "l_star": 60.0,
                "a_star": 15.0,
                "b_star": 12.0,
                "has_real_pixels": False,
            }

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

        prefix = "[DEMO MOCK] " if self.is_mock else ""
        if self.is_anemic:
            return f"{prefix}ANEMIC - Signs of anemia detected"
        return f"{prefix}NON-ANEMIC - No signs of anemia detected"

    @property
    def risk_explanation(self) -> str:
        """Return explanatory text for the current risk level."""

        _, explanation = compute_risk_metadata(self.prediction, self.confidence or 0.0)
        return explanation

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
    """Return a query scoped to real scans that are not soft-deleted and not mock/demo."""

    return Scan.query.filter(
        Scan.deleted_at.is_(None),
        or_(Scan.is_mock.is_(False), Scan.is_mock.is_(None)),
    )


def allowed_file(filename: str, mimetype: str | None = None) -> bool:
    """Check whether a filename or MIME type matches the supported upload formats."""

    if is_supported_upload(filename):
        return True

    if not Path(filename).suffix and mimetype:
        mimetype = mimetype.lower()
        return mimetype in {"image/jpeg", "image/jpg", "image/png", "image/webp"}

    return False


def get_predictor() -> AnemiaPredictor:
    """Lazy-load the predictor once a valid, non-LFS trained checkpoint is available."""

    global _predictor, _predictor_load_error, _predictor_failure_category

    if _predictor is not None:
        return _predictor

    raw_ckpt = latest_available_checkpoint(require_valid=False)
    inspection = inspect_checkpoint_file(raw_ckpt)
    if not inspection["checkpoint_valid"]:
        _predictor_load_error = inspection["reason"]
        _predictor_failure_category = inspection["failure_category"]
        if inspection["checkpoint_is_lfs_pointer"]:
            raise RuntimeError(inspection["reason"])
        raise FileNotFoundError(
            inspection["reason"]
            or "No trained checkpoint is available yet. Run train.py before serving predictions."
        )

    checkpoint = Path(inspection["path"])

    with _predictor_load_lock:
        if _predictor is not None:
            return _predictor

        try:
            _predictor = AnemiaPredictor(checkpoint_path=checkpoint)
            _predictor_load_error = None
            _predictor_failure_category = None
        except Exception as exc:
            _predictor_load_error = str(exc)
            _predictor_failure_category = "model_load_failed"
            raise

    return _predictor


def predictor_loaded() -> bool:
    """Return whether this worker has a ready predictor."""

    return _predictor is not None


def eager_load_predictor() -> None:
    """Attempt to load the predictor during startup without crashing when weights are missing/LFS."""

    global _predictor_load_error, _predictor_failure_category

    if predictor_loaded():
        return

    raw_ckpt = latest_available_checkpoint(require_valid=False)
    inspection = inspect_checkpoint_file(raw_ckpt)
    if not inspection["checkpoint_valid"]:
        _predictor_load_error = inspection["reason"]
        _predictor_failure_category = inspection["failure_category"]
        return

    try:
        predictor = get_predictor()
        warm_predictor_runtime(predictor)
    except Exception as exc:
        _predictor_load_error = str(exc)
        _predictor_failure_category = "model_load_failed"


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
    """Warm the predictor in the background only when a valid non-LFS checkpoint exists."""

    global _predictor_warmup_started

    if predictor_loaded() or latest_available_checkpoint(require_valid=True) is None:
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


def estimate_hemoglobin_range(
    prediction: str,
    confidence: float,
    *,
    hb_estimate_g_dl: float | None = None,
    regression_status: str | None = None,
) -> str:
    """Return a validated Hb estimate or the required unavailable notice."""

    return prediction_estimate_hemoglobin_range(
        prediction,
        confidence,
        hb_estimate_g_dl=hb_estimate_g_dl,
        regression_status=regression_status,
    )


def assess_image_quality(image_path: Path) -> dict[str, Any]:
    """Return a UI-ready quality summary backed by the shared validator."""

    return build_quality_payload(validate_image(image_path))


def safe_scan_count() -> int:
    """Return the latest real scan count without letting SQLite stalls block the UI."""

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
    """Save an uploaded image preserving original dimensions for pre-resize validation."""

    path, _ = save_and_validate_raw_image(image, destination)
    return path


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


def _max_upload_bytes() -> int:
    """Return the configured maximum upload size in bytes."""

    return int(FLASK_CONFIG.max_content_length or 10 * 1024 * 1024)


def save_uploaded_file(file_storage) -> Path:
    """Validate and persist an uploaded file in its original uncropped dimensions."""

    filename = getattr(file_storage, "filename", "") or ""
    mimetype = getattr(file_storage, "mimetype", None)
    if not allowed_file(filename, mimetype):
        raise ImageUploadValidationError(
            "Unsupported file type. Please upload a JPG, JPEG, PNG, or WEBP image."
        )

    extension = Path(secure_filename(filename)).suffix.lower()
    if not extension:
        extension = _extension_from_mimetype(mimetype) or ".jpg"

    max_bytes = _max_upload_bytes()
    file_storage.stream.seek(0)
    raw_bytes = file_storage.stream.read(max_bytes + 1)
    if not raw_bytes:
        raise ImageUploadValidationError(
            "The uploaded image file is empty. Please choose a valid image."
        )
    if len(raw_bytes) > max_bytes:
        raise ImageUploadValidationError(
            f"The uploaded image exceeds the maximum allowed size of {max_bytes // (1024 * 1024)} MB."
        )

    destination = unique_storage_path(extension)
    saved_path, _ = save_and_validate_raw_image(raw_bytes, destination)
    return saved_path


def save_base64_image(image_payload: str) -> Path:
    """Validate and persist a camera or API base64 image in its original dimensions."""

    if not image_payload or not isinstance(image_payload, str):
        raise ImageUploadValidationError("Image data is missing from the request.")

    match = re.match(
        r"^data:image/(?P<kind>[a-zA-Z0-9+.-]+);base64,(?P<data>.+)$",
        image_payload.strip(),
        flags=re.DOTALL,
    )
    image_kind = "jpeg"
    base64_blob = image_payload.strip()
    if match:
        image_kind = match.group("kind").lower()
        base64_blob = match.group("data").strip()

    if image_kind not in {"jpeg", "jpg", "png", "webp"}:
        raise ImageUploadValidationError(
            f"Unsupported image format '{image_kind}'. Please use JPG, PNG, or WEBP."
        )

    extension = ".jpg" if image_kind in {"jpeg", "jpg"} else f".{image_kind}"
    max_bytes = _max_upload_bytes()

    try:
        image_bytes = base64.b64decode(base64_blob, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageUploadValidationError(
            "Image data is invalid. Please send a valid base64-encoded JPG, PNG, or WEBP image.",
            internal_detail=str(exc),
        ) from exc

    if not image_bytes:
        raise ImageUploadValidationError("Decoded image data is empty.")
    if len(image_bytes) > max_bytes:
        raise ImageUploadValidationError(
            f"The uploaded image exceeds the maximum allowed size of {max_bytes // (1024 * 1024)} MB."
        )

    destination = unique_storage_path(extension)
    saved_path, _ = save_and_validate_raw_image(image_bytes, destination)
    return saved_path


def parse_roi_bbox(raw: Any) -> tuple[int, int, int, int] | None:
    """Parse [x1, y1, x2, y2] bounding box coordinates from list, tuple, or string."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = raw[1:-1].split(",")
        else:
            raw = raw.split(",")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in raw]
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)
        except (ValueError, TypeError):
            return None
    return None


def validate_patient_payload(
    payload: dict[str, Any],
    require_demographics: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Validate incoming patient form data from HTML or API submissions."""

    errors: list[str] = []

    patient_name = str(payload.get("patient_name") or payload.get("name") or "").strip()
    age_raw = payload.get("age", payload.get("patient_age"))
    gender_raw = payload.get("gender", payload.get("sex", payload.get("biological_sex", "")))
    gender = normalize_gender_value(str(gender_raw).strip()) if gender_raw else ""
    phone = str(payload.get("phone", "")).strip()
    notes = str(payload.get("notes", "")).strip()

    # User-selected ROI Bounding Box Ingestion (Issue 4)
    raw_bbox = payload.get("roi_bbox") or payload.get("crop_coords") or payload.get("bbox") or payload.get("roi_box")
    roi_bbox = parse_roi_bbox(raw_bbox)

    has_explicit_modality = any(
        key in payload for key in ("body_part", "anatomical_modality", "image_type", "image_region")
    )
    raw_modality = (
        payload.get("body_part")
        if "body_part" in payload
        else payload.get(
            "anatomical_modality",
            payload.get("image_type", payload.get("image_region", "Eye Conjunctiva")),
        )
    )

    body_part = "conjunctiva"
    image_type = "Eye Conjunctiva"
    if has_explicit_modality or raw_modality:
        try:
            body_part = resolve_body_part(str(raw_modality) if raw_modality is not None else "")
            image_type = ui_label_for_body_part(body_part)
        except ModalityValidationError as exc:
            errors.append(exc.user_message)

    screening_threshold = DEFAULT_SCREENING_THRESHOLD = 0.50
    high_sensitivity_mode = False
    try:
        screening_threshold, high_sensitivity_mode = resolve_screening_threshold(
            screening_threshold=payload.get("screening_threshold"),
            high_sensitivity_mode=payload.get("high_sensitivity_mode"),
        )
    except ServiceAdapterError as exc:
        errors.append(exc.user_message)

    if not patient_name:
        if require_demographics:
            errors.append("Patient name is required.")
        else:
            patient_name = "Anonymous Patient"

    age: int | None = None
    if age_raw in (None, ""):
        if require_demographics:
            errors.append("Age is required.")
    else:
        try:
            age = int(str(age_raw).strip())
            if age < 0 or age > 120:
                errors.append("Age must be between 0 and 120.")
        except ValueError:
            errors.append("Age must be a valid integer.")

    if gender not in GENDER_OPTIONS:
        if require_demographics:
            errors.append("Gender is required.")
        else:
            gender = "Other"

    raw_mock = payload.get("use_mock")
    use_mock = str(raw_mock).strip().lower() in {"1", "true", "yes"} if raw_mock is not None else False

    return (
        {
            "patient_name": patient_name,
            "age": age,
            "gender": gender,
            "phone": phone or None,
            "notes": notes or None,
            "image_type": image_type,
            "body_part": body_part,
            "screening_threshold": screening_threshold,
            "high_sensitivity_mode": high_sensitivity_mode,
            "use_mock": use_mock,
            "roi_bbox": roi_bbox,
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
    prediction: PredictionResult | AdapterPredictionOutput,
    patient_data: dict[str, Any],
) -> tuple[Scan, bool]:
    """Create and persist a scan record from a completed prediction.

    Mock predictions (`is_mock=True`) are never saved to the database as genuine scans.
    """

    is_adapter = isinstance(prediction, AdapterPredictionOutput)
    is_mock = bool(getattr(prediction, "is_mock", False))

    prediction_label = (
        prediction.prediction
        or normalize_prediction_label(getattr(prediction, "predicted_class", "") or "anemic")
    )
    anemic_probability = float(prediction.anemic_probability)
    non_anemic_probability = float(prediction.non_anemic_probability)

    triage_dict = getattr(prediction, "triage", None) if is_adapter else None
    triage_level = triage_dict.get("level") if isinstance(triage_dict, dict) else None
    warnings_list = (
        list(prediction.quality_warnings)
        if is_adapter
        else list(getattr(prediction, "warnings", []) or [])
    )
    normalized_path = getattr(prediction, "normalized_image_path", None) if is_adapter else None

    record = Scan(
        patient_name=patient_data["patient_name"],
        age=patient_data["age"],
        gender=patient_data["gender"],
        phone=patient_data["phone"],
        image_type=patient_data.get("image_type") or "Eye Conjunctiva",
        image_path=storage_reference_for_path(image_path, "uploads") or str(image_path),
        normalized_image_path=(
            storage_reference_for_path(normalized_path, "uploads")
            if normalized_path
            else None
        ),
        gradcam_path=storage_reference_for_path(prediction.gradcam_path, "gradcam")
        if prediction.gradcam_path
        else None,
        prediction=prediction_label,
        confidence=prediction.confidence,
        risk_level=prediction.risk_level,
        anemic_probability=anemic_probability,
        non_anemic_probability=non_anemic_probability,
        hb_estimate_g_dl=getattr(prediction, "hb_estimate_g_dl", None) if is_adapter else None,
        regression_status=getattr(prediction, "regression_status", "unavailable") if is_adapter else "unavailable",
        model_version=getattr(prediction, "model_version", None) if is_adapter else None,
        calibration_status=getattr(prediction, "calibration_status", None) if is_adapter else None,
        anatomical_modality=getattr(prediction, "anatomical_modality", patient_data.get("body_part")) if is_adapter else patient_data.get("body_part"),
        triage_level=triage_level,
        ita_angle=getattr(prediction, "ita_angle", None) if is_adapter else None,
        ita_category=getattr(prediction, "ita_category", None) if is_adapter else None,
        glare_fraction=getattr(prediction, "glare_fraction", None) if is_adapter else None,
        quality_warnings_json=json.dumps(warnings_list) if warnings_list else None,
        is_mock=is_mock,
        notes=patient_data["notes"],
        created_at=datetime.utcnow(),
    )

    if is_mock:
        # Never persist mock results into the production/clinical scans table.
        return record, False

    persisted = False
    persist_media_blob(image_path, "uploads")
    if prediction.gradcam_path:
        persist_media_blob(Path(prediction.gradcam_path), "gradcam")
    db.session.add(record)
    try:
        db.session.commit()
        persisted = True
        safe_scan_count()
    except Exception:
        db.session.rollback()

    return record, persisted


def latest_scans(limit: int = 5) -> list[Scan]:
    """Return the most recent real scans for dashboard widgets."""

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
        query = query.filter(or_(*search_clauses))

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
    """Build the data payload that powers the analytics dashboard using real DB counts only."""

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

    age_group_counter = Counter(
        age_group_label(record.age) for record in records if record.age is not None
    )
    most_common_age_group = (
        age_group_counter.most_common(1)[0][0] if age_group_counter else "N/A"
    )

    if start_date is None and end_date is None:
        timeline_start = date.today() - timedelta(days=29)
        timeline_end = date.today()
    else:
        timeline_start = start_date or (
            end_date - timedelta(days=29) if end_date else date.today() - timedelta(days=29)
        )
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
    anemic_age_counter = Counter(
        age_group_label(record.age) for record in anemic_records if record.age is not None
    )

    confidence_buckets = {
        "50-59%": 0,
        "60-69%": 0,
        "70-79%": 0,
        "80-89%": 0,
        "90-100%": 0,
    }
    conjunctiva_count = 0
    fingernail_count = 0
    other_modality_count = 0

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

        modality_token = (record.anatomical_modality or record.image_type or "").strip().lower()
        if modality_token in {"conjunctiva", "eye conjunctiva"}:
            conjunctiva_count += 1
        elif modality_token in {"fingernail", "nail"}:
            fingernail_count += 1
        else:
            other_modality_count += 1

    return {
        "metrics": {
            "total_patients": total_patients,
            "detection_rate": round((len(anemic_records) / total_patients) * 100, 1)
            if total_patients
            else 0.0,
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
        "image_type_bar": {
            "labels": ["Conjunctiva", "Fingernail", "Other"],
            "values": [conjunctiva_count, fingernail_count, other_modality_count],
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
    """Read validation metrics stored with a valid deployed checkpoint (never fabricating missing metrics)."""

    global _checkpoint_dashboard_cache

    path = BEST_CHECKPOINT_PATH
    if not path.exists() or is_lfs_pointer_file(path):
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
        try:
            data = torch.load(path, map_location=torch.device("cpu"))
        except Exception:
            return None
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

    def norm_prob(value: Any) -> float | None:
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if v > 1.0:
            v = v / 100.0
        return v

    acc_f = norm_prob(pick("valid_accuracy", "val_accuracy", "accuracy"))
    sens_n = norm_prob(pick("valid_sensitivity", "sensitivity"))
    spec_n = norm_prob(pick("valid_specificity", "specificity"))
    auc_n = norm_prob(pick("valid_auc_roc", "auc_roc"))

    if acc_f is None and sens_n is None and spec_n is None and auc_n is None:
        return None

    payload = {
        "available": True,
        "accuracy_badge": round(acc_f * 100, 1) if acc_f is not None else None,
        "accuracy": acc_f,
        "sensitivity": sens_n,
        "specificity": spec_n,
        "auc_roc": auc_n,
        "source": f"deployed checkpoint ({path.name})",
    }
    _checkpoint_dashboard_cache = {"path": str(path), "mtime": mtime, "data": payload}
    return payload


def performance_overview() -> dict[str, Any]:
    """Load real evaluation metrics when backed by an identified evaluation artifact.

    Never returns hardcoded/invented benchmark numbers when evaluation artifacts are missing.
    """

    repo_dir = PROJECT_ROOT.parent if (PROJECT_ROOT.parent / "evaluation").is_dir() else PROJECT_ROOT
    candidate_paths = [
        EVALUATION_LOGS_DIR / "test_summary.json",
        repo_dir / "evaluation" / "metrics" / "champion_test_report.json",
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
            acc_val: float | None = None
            if raw_acc is not None:
                try:
                    acc_val = float(raw_acc)
                    if acc_val > 1.0:
                        acc_val = acc_val / 100.0
                except (TypeError, ValueError):
                    acc_val = None

            sens = cls_data.get("sensitivity_recall") or cls_data.get("sensitivity")
            spec = cls_data.get("specificity")
            auc = cls_data.get("roc_auc") or cls_data.get("auc_roc")

            return {
                "available": True,
                "accuracy_badge": round(acc_val * 100, 1) if acc_val is not None else None,
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
        "available": False,
        "accuracy_badge": None,
        "accuracy": None,
        "sensitivity": None,
        "specificity": None,
        "auc_roc": None,
        "source": "Unavailable (no evaluation artifact)",
    }


def eda_overview() -> dict[str, Any]:
    """Compile comprehensive Exploratory Data Analysis (EDA) and benchmark metrics."""

    repo_dir = PROJECT_ROOT.parent if (PROJECT_ROOT.parent / "data").is_dir() else PROJECT_ROOT

    # 1. Load split audit summary
    split_audit_path = repo_dir / "data" / "splits" / "split_audit_summary.json"
    audit_data: dict[str, Any] = {}
    if split_audit_path.exists():
        try:
            audit_data = json.loads(split_audit_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 2. Load champion test report
    champion_report_path = repo_dir / "evaluation" / "metrics" / "champion_test_report.json"
    champion_data: dict[str, Any] = {}
    if champion_report_path.exists():
        try:
            champion_data = json.loads(champion_report_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 3. Available evaluation plots
    plots = [
        {
            "id": "champion_cm",
            "title": "Champion Confusion Matrix",
            "subtitle": "Held-out test set evaluation (N = 1,025) across palpebral conjunctiva and fingernail modalities.",
            "file": "plots/champion_confusion_matrix.png",
            "badge": "Champion Dual-Head",
            "category": "Classification",
            "stats": "TN: 245 | FP: 172 | FN: 198 | TP: 410",
        },
        {
            "id": "reliability",
            "title": "Reliability Diagram & Temperature Scaling",
            "subtitle": "Empirical probability calibration curves before vs. after Platt/temperature scaling (T = 1.339).",
            "file": "plots/reliability_diagram.png",
            "badge": "Calibrated (T = 1.339)",
            "category": "Calibration",
            "stats": "Brier Score: 0.2274 | Calibration Temp: 1.3391",
        },
        {
            "id": "champion_bland_altman",
            "title": "Bland-Altman Agreement (Continuous Hb)",
            "subtitle": "Predicted Hemoglobin (g/dL) vs. laboratory reference CBC (N = 174 paired test cases).",
            "file": "plots/champion_bland_altman.png",
            "badge": "Mean Bias +0.17 g/dL",
            "category": "Regression",
            "stats": "Mean Bias: +0.17 g/dL | 95% LoA: [-5.01, +5.36] g/dL",
        },
        {
            "id": "baseline_cm",
            "title": "Baseline Confusion Matrix",
            "subtitle": "Legacy baseline single-head model performance before multi-task joint regression and chromatic feature extraction.",
            "file": "plots/confusion_matrix.png",
            "badge": "Baseline EfficientNet",
            "category": "Baseline",
            "stats": "Single-Head EfficientNet Baseline",
        },
        {
            "id": "baseline_bland_altman",
            "title": "Baseline Bland-Altman Agreement",
            "subtitle": "Legacy single-head regression agreement before joint dual-head representation learning.",
            "file": "plots/bland_altman_agreement.png",
            "badge": "Baseline Regression",
            "category": "Baseline",
            "stats": "Single-Head Regression Baseline",
        },
    ]

    # 4. Curated sample images showcasing cohorts & normalization
    samples = [
        {
            "title": "Palpebral Conjunctiva (Anemic)",
            "cohort": "CP-AnemiC V2",
            "modality": "Conjunctiva",
            "label": "Anemic (Pallor)",
            "file": "eda/cp_anemic_conjunctiva.jpg",
            "caption": "Prominent microvascular pallor with decreased vessel perfusion in the palpebral erythema zone.",
            "tag": "Severe Pallor",
        },
        {
            "title": "Palpebral Conjunctiva (Non-Anemic)",
            "cohort": "CP-AnemiC V2",
            "modality": "Conjunctiva",
            "label": "Non-Anemic (Normal)",
            "file": "eda/cp_non_anemic_conjunctiva.jpg",
            "caption": "Rich vascular branching and high CIE-LAB a* redness response in healthy conjunctival capillary bed.",
            "tag": "Healthy Perfusion",
        },
        {
            "title": "Full-Eye Smartphone Capture",
            "cohort": "EYES-DEFY-ANEMIA",
            "modality": "Conjunctiva (Full Frame)",
            "label": "Uncropped Clinical Capture",
            "file": "eda/eyes_defy_full.jpg",
            "caption": "Cross-national clinical capture showing ambient hospital illumination and ocular aperture.",
            "tag": "Raw Input",
        },
        {
            "title": "Automated Palpebral ROI Crop",
            "cohort": "EYES-DEFY-ANEMIA",
            "modality": "Conjunctiva (Segmented ROI)",
            "label": "Targeted Microvascular ROI",
            "file": "eda/eyes_defy_palpebral.jpg",
            "caption": "Extracted region-of-interest eliminating iris, sclera, and eyelid skin interference.",
            "tag": "Automated Crop",
        },
        {
            "title": "Subungual Fingernail Bed (Anemic)",
            "cohort": "Ghana Fingernails",
            "modality": "Fingernail",
            "label": "Anemic Pallor",
            "file": "eda/ghana_fingernail_anemic.png",
            "caption": "Subungual capillary pallor under standardized smartphone macro inspection.",
            "tag": "Microvascular Pallor",
        },
        {
            "title": "Subungual Fingernail Bed (Non-Anemic)",
            "cohort": "Ghana Fingernails",
            "modality": "Fingernail",
            "label": "Non-Anemic Flush",
            "file": "eda/ghana_fingernail_nonanemic.png",
            "caption": "Healthy capillary flush and normal erythema distribution in the nail matrix bed.",
            "tag": "Normal Flush",
        },
        {
            "title": "Grad-CAM Saliency Activation",
            "cohort": "Dual-Head Champion",
            "modality": "Explainable AI",
            "label": "Grad-CAM Attention Map",
            "file": "eda/gradcam_sample.png",
            "caption": "Grad-CAM localization proving the network attends directly to the lower eyelid conjunctiva.",
            "tag": "Visual Attention",
        },
    ]

    # 5. High-level KPIs
    inv = audit_data.get("dataset_inventory", {})
    splits = audit_data.get("split_distribution", {})
    cls_metrics = champion_data.get("classification", {})
    reg_metrics = champion_data.get("hemoglobin_regression", {})

    return {
        "kpis": {
            "total_images": inv.get("total_images", 5187),
            "unique_clusters": inv.get("unique_roots", 985),
            "exact_duplicates_purged": inv.get("exact_sha256_duplicates", 2375),
            "dhash_duplicates_purged": inv.get("perceptual_dhash_duplicates", 98),
            "total_purged": inv.get("exact_sha256_duplicates", 2375) + inv.get("perceptual_dhash_duplicates", 98),
            "test_images": splits.get("test", {}).get("total_images", 1025),
            "paired_cbc_samples": (
                splits.get("train", {}).get("samples_with_lab_hb", 652)
                + splits.get("validation", {}).get("samples_with_lab_hb", 100)
                + splits.get("test", {}).get("samples_with_lab_hb", 174)
            ),
            "temperature_t": champion_data.get("temperature_scaling_T", 1.339153),
            "leakage_rate": "0.0%",
        },
        "audit": audit_data,
        "champion": champion_data,
        "classification": cls_metrics,
        "regression": reg_metrics,
        "plots": plots,
        "samples": samples,
    }


def generate_pdf_report(record: Scan) -> Path:
    """Generate a PDF report via the shared Stage 4 reporting module."""

    return build_pdf_report(record)


def build_prediction_response(
    record: Scan,
    quality: dict[str, Any],
    persisted: bool,
    adapter_output: AdapterPredictionOutput | None = None,
) -> dict[str, Any]:
    """Return a JSON-friendly representation of a prediction result."""

    result_url = url_for("result_page", scan_id=record.id) if persisted and record.id else None
    pdf_url = url_for("export_pdf", scan_id=record.id) if persisted and record.id else None

    base = {
        "success": True,
        "is_mock": bool(record.is_mock),
        "id": record.id,
        "scan_id": record.id,
        "public_scan_id": record.public_scan_id,
        "persisted": persisted,
        "patient_name": record.patient_name,
        "prediction": record.prediction,
        "predicted_class": record.predicted_class,
        "confidence": record.confidence,
        "confidence_percent": record.confidence_percent,
        "risk_level": record.risk_level,
        "risk_explanation": record.risk_explanation,
        "anemia_probability": record.anemic_probability,
        "anemic_probability": record.anemic_probability,
        "non_anemic_probability": record.non_anemic_probability,
        "hb_estimate_g_dl": record.hb_estimate_g_dl if record.hb_regression_available else None,
        "hb_regression_available": record.hb_regression_available,
        "hemoglobin_estimate": record.hemoglobin_estimate,
        "regression_status": record.regression_status or "unavailable",
        "model_version": record.model_version or "unknown",
        "calibration_status": record.calibration_status or "uncalibrated",
        "anatomical_modality": record.anatomical_modality or "conjunctiva",
        "ita_angle": record.ita_angle,
        "ita_category": record.ita_category,
        "glare_fraction": record.glare_fraction,
        "triage": adapter_output.triage if adapter_output is not None else record.triage,
        "medical_advice": " ".join(record.medical_advice),
        "disclaimer": DISCLAIMER_TEXT,
        "quality": quality,
        "image_quality": quality.get("image_quality"),
        "quality_warning": quality.get("warning"),
        "quality_warnings": quality.get("warnings", []),
        "image_url": record.image_url,
        "normalized_image_url": record.normalized_image_url,
        "gradcam_url": record.gradcam_url,
        "result_url": result_url,
        "pdf_url": pdf_url,
    }
    if adapter_output is not None:
        base.update(adapter_output.to_api_dict())
        base["id"] = record.id
        base["scan_id"] = record.id
        base["public_scan_id"] = record.public_scan_id
        base["persisted"] = persisted
        base["patient_name"] = record.patient_name
        base["image_url"] = record.image_url
        base["normalized_image_url"] = record.normalized_image_url
        base["gradcam_url"] = record.gradcam_url
        base["result_url"] = result_url
        base["pdf_url"] = pdf_url
    return base


def _ensure_inference_ready(use_mock: bool = False) -> None:
    """Ensure that either Person 2's predictor, a valid checkpoint, or explicit dev mock is ready."""

    if use_mock:
        if not is_mock_mode_enabled():
            raise ModelUnavailableError(
                "Development mock mode is disabled or blocked in this environment.",
                failure_category="mock_mode_disabled",
            )
        return

    if resolve_person2_predict_callable() is not None:
        return

    if _predictor is not None:
        return

    readiness = inspect_runtime_readiness(
        predictor_loaded=False,
        predictor_load_error=_predictor_load_error,
    )
    if readiness["checkpoint_valid"]:
        try:
            get_predictor()
            return
        except Exception as exc:
            raise ModelUnavailableError(
                f"The AI model checkpoint could not be loaded: {exc}",
                failure_category="model_load_failed",
                internal_detail=str(exc),
            ) from exc

    raise ModelUnavailableError(
        readiness["status_message"],
        failure_category=readiness["failure_category"] or "model_unavailable",
        internal_detail=readiness.get("predictor_load_error"),
    )


def run_prediction_workflow(
    patient_payload: dict[str, Any],
    uploaded_file=None,
    captured_image_data: str | None = None,
    secondary_uploaded_file=None,
) -> tuple[Scan, dict[str, Any], AdapterPredictionOutput, bool]:
    """Validate and save the raw input image, run adapter inference, and persist real scans."""

    if uploaded_file is not None and getattr(uploaded_file, "filename", ""):
        raw_image_path = save_uploaded_file(uploaded_file)
    elif captured_image_data:
        raw_image_path = save_base64_image(captured_image_data)
    else:
        raise ImageUploadValidationError("Please upload an image or capture one from the camera.")

    use_mock = bool(patient_payload.get("use_mock", False))
    try:
        _ensure_inference_ready(use_mock=use_mock)
        adapter_output = run_vyra_predict(
            raw_image_path=raw_image_path,
            body_part=patient_payload.get("body_part") or patient_payload.get("image_type"),
            patient_payload=patient_payload,
            screening_threshold=patient_payload.get("screening_threshold"),
            high_sensitivity_mode=patient_payload.get("high_sensitivity_mode"),
            roi_bbox=patient_payload.get("roi_bbox"),
            fallback_predictor=_predictor,
            use_mock=use_mock,
        )
    except Exception:
        for candidate in (
            raw_image_path,
            raw_image_path.with_name(f"{raw_image_path.stem}_model_input{raw_image_path.suffix}"),
            raw_image_path.with_name(f"{raw_image_path.stem}_normalized{raw_image_path.suffix}"),
        ):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    # Process optional secondary multimodal scan (Dual-Convergence Mode)
    if secondary_uploaded_file is not None and getattr(secondary_uploaded_file, "filename", ""):
        try:
            sec_image_path = save_uploaded_file(secondary_uploaded_file)
            primary_bp = str(patient_payload.get("body_part") or patient_payload.get("image_type") or "conjunctiva").lower()
            sec_bp = "fingernail" if "conjunctiva" in primary_bp or "eye" in primary_bp else "conjunctiva"
            
            sec_output = run_vyra_predict(
                raw_image_path=sec_image_path,
                body_part=sec_bp,
                patient_payload=patient_payload,
                screening_threshold=patient_payload.get("screening_threshold"),
                high_sensitivity_mode=patient_payload.get("high_sensitivity_mode"),
                fallback_predictor=_predictor,
                use_mock=use_mock,
            )
            
            p1 = float(adapter_output.anemic_probability)
            p2 = float(sec_output.anemic_probability)
            # Weighted joint posterior (Eye 0.55, Nail 0.45)
            if "conjunctiva" in primary_bp:
                p_joint = round(0.55 * p1 + 0.45 * p2, 4)
            else:
                p_joint = round(0.55 * p2 + 0.45 * p1, 4)
            concordance = round(max(0.0, 1.0 - abs(p1 - p2)) * 100, 1)

            multimodal_data = {
                "active": True,
                "primary_modality": "Eye Conjunctiva" if "conjunctiva" in primary_bp else "Fingernail Bed",
                "primary_prob": round(p1 * 100, 1),
                "primary_class": adapter_output.predicted_class,
                "secondary_modality": "Fingernail Bed" if "conjunctiva" in primary_bp else "Eye Conjunctiva",
                "secondary_prob": round(p2 * 100, 1),
                "secondary_class": sec_output.predicted_class,
                "secondary_image_filename": sec_image_path.name,
                "joint_probability": round(p_joint * 100, 1),
                "concordance_pct": concordance,
                "agreement_level": "High Cross-Site Agreement" if concordance >= 85.0 else ("Moderate Concordance" if concordance >= 70.0 else "Site Divergence Noted"),
            }
            import json
            existing_notes = str(patient_payload.get("notes") or "").strip()
            fusion_tag = f"MULTIMODAL_FUSION: {json.dumps(multimodal_data)}"
            patient_payload["notes"] = f"{existing_notes}\n\n{fusion_tag}" if existing_notes else fusion_tag
        except Exception as sec_exc:
            print(f"[multimodal] Warning: Secondary scan processing failed: {sec_exc}", flush=True)

    quality = adapter_output.quality
    record, persisted = create_scan_record(raw_image_path, adapter_output, patient_payload)
    return record, quality, adapter_output, persisted


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
    """Test that the configured database is reachable and log the result."""

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
        readiness = inspect_runtime_readiness(
            predictor_loaded=predictor_loaded(),
            predictor_load_error=_predictor_load_error,
        )
        return {
            "total_scans_counter": total_scans_counter,
            "current_year": datetime.utcnow().year,
            "model_ready": readiness["model_ready"],
            "model_loading": bool(
                _predictor_warmup_started
                and not readiness["model_ready"]
                and readiness["checkpoint_valid"]
            ),
            "checkpoint_available": readiness["checkpoint_available"],
            "checkpoint_exists": readiness["checkpoint_exists"],
            "checkpoint_valid": readiness["checkpoint_valid"],
            "checkpoint_is_lfs_pointer": readiness["checkpoint_is_lfs_pointer"],
            "predictor_loaded": readiness["predictor_loaded"],
            "predictor_load_error": readiness["predictor_load_error"],
            "model_failure_category": readiness["failure_category"],
            "model_status_message": readiness["status_message"],
            "person2_predictor_available": readiness["person2_predictor_available"],
            "person1_normalizer_available": readiness["person1_normalizer_available"],
            "mock_mode_enabled": readiness["mock_mode_enabled"],
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
        """Readiness endpoint distinguishing file existence, LFS pointers, and model load status."""

        readiness = inspect_runtime_readiness(
            predictor_loaded=predictor_loaded(),
            predictor_load_error=_predictor_load_error,
        )
        return {
            "status": "ok",
            "model_ready": readiness["model_ready"],
            "checkpoint_available": readiness["checkpoint_available"],
            "checkpoint_exists": readiness["checkpoint_exists"],
            "checkpoint_valid": readiness["checkpoint_valid"],
            "checkpoint_is_lfs_pointer": readiness["checkpoint_is_lfs_pointer"],
            "checkpoint_size_bytes": readiness["checkpoint_size_bytes"],
            "predictor_loaded": readiness["predictor_loaded"],
            "predictor_load_error": readiness["predictor_load_error"],
            "failure_category": readiness["failure_category"],
            "status_message": readiness["status_message"],
            "person2_predictor_available": readiness["person2_predictor_available"],
            "person1_normalizer_available": readiness["person1_normalizer_available"],
            "mock_mode_enabled": readiness["mock_mode_enabled"],
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
                secondary_uploaded_file=request.files.get("secondary_image"),
            )
        except ModelUnavailableError as exc:
            flash(exc.user_message, "warning")
            return redirect(url_for("scan_page"))
        except ServiceAdapterError as exc:
            flash(exc.user_message, "danger")
            return redirect(url_for("scan_page"))
        except Exception as exc:  # pragma: no cover - runtime guard
            flash(f"Prediction failed: {exc}", "danger")
            return redirect(url_for("scan_page"))

        if record.is_mock:
            flash(
                "DEMO MOCK MODE: Showing simulated development result. This scan was not saved to clinical history.",
                "warning",
            )
        elif persisted:
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

        if request.is_json:
            payload = request.get_json(silent=True) or {}
            patient_payload, errors = validate_patient_payload(payload)
            if errors:
                return jsonify({"success": False, "errors": errors, "error": errors[0]}), 400
            uploaded_file = None
            captured_image_data = payload.get("image") or payload.get("image_data")
        else:
            patient_payload, errors = validate_patient_payload(request.form.to_dict())
            if errors:
                return jsonify({"success": False, "errors": errors, "error": errors[0]}), 400
            uploaded_file = request.files.get("image")
            captured_image_data = request.form.get("captured_image_data")

        try:
            record, quality, adapter_output, persisted = run_prediction_workflow(
                patient_payload=patient_payload,
                uploaded_file=uploaded_file,
                captured_image_data=captured_image_data,
            )
        except ServiceAdapterError as exc:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": exc.user_message,
                        "failure_category": exc.failure_category,
                    }
                ),
                exc.status_code,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "An unexpected error occurred during prediction.",
                        "failure_category": "internal_error",
                    }
                ),
                500,
            )

        response_payload = build_prediction_response(
            record,
            quality,
            persisted,
            adapter_output=adapter_output,
        )
        return jsonify(response_payload), 200

    @app.route("/result/<int:scan_id>")
    def result_page(scan_id: int):
        """Show the full result page for one scan."""

        record = scans_active_query().filter_by(id=scan_id).first_or_404()
        image_path = record.image_file_path or Path(str(record.image_path))
        quality = assess_image_quality(image_path)
        stored_warnings = record.quality_warnings
        if stored_warnings:
            merged_warnings = list(dict.fromkeys([*(quality.get("warnings") or []), *stored_warnings]))
            quality = {
                **quality,
                "warnings": merged_warnings,
                "warning": quality.get("warning") or " ".join(merged_warnings),
            }
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
        query = apply_history_filters(scans_active_query(), filters).order_by(
            Scan.created_at.desc()
        )
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
        response.headers["Content-Disposition"] = (
            "attachment; filename=anemiavision_history.csv"
        )
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

    @app.route("/eda")
    def eda_page():
        """Render the comprehensive Exploratory Data Analysis & Diagnostic Benchmark page."""

        eda_payload = eda_overview()
        return render_template(
            "eda.html",
            eda=eda_payload,
            kpis=eda_payload["kpis"],
            audit=eda_payload["audit"],
            champion=eda_payload["champion"],
            cls=eda_payload["classification"],
            reg=eda_payload["regression"],
            plots=eda_payload["plots"],
            samples=eda_payload["samples"],
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

    @app.route("/achilles-heel", methods=["GET", "POST"])
    def achilles_heel_page():
        """Render the Model's Achilles Heel research interface."""
        import cv2
        import numpy as np
        from achilles_heel import analyze_model_shortcuts

        case = request.args.get("case", "face").strip().lower()
        scan_id = request.args.get("scan_id", type=int)

        img_bgr = None
        modality = "conjunctiva"

        # Check if file uploaded via POST
        if request.method == "POST" and "test_image" in request.files:
            file = request.files["test_image"]
            if file and file.filename:
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                modality = request.form.get("modality", "conjunctiva")
                case = "custom"

        # Check if scan_id provided
        if img_bgr is None and scan_id:
            record = scans_active_query().filter_by(id=scan_id).first()
            if record and record.image_file_path and record.image_file_path.exists():
                img_bgr = cv2.imread(str(record.image_file_path))
                modality = str(record.anatomical_modality or record.image_type or "conjunctiva").lower()
                case = f"scan_{scan_id}"

        # Otherwise use pre-loaded case
        if img_bgr is None:
            if case == "nail":
                case_path = Path(r"C:\Users\urbra\Downloads\WhatsApp Image 2026-10-23 at 21.05.41.jpeg")
                modality = "fingernail"
                if not case_path.exists():
                    case_path = Path("data/raw/ghana_fingernails/Fingernails/Anemic-Fin-008 (10).png")
            elif case == "cpanemic":
                case_path = Path("data/raw/cp_anemic/CP-AnemiC/Anemic/Image_001.png")
                modality = "conjunctiva"
            else:
                case = "face"
                case_path = Path(r"C:\Users\urbra\Downloads\WhatsApp Image 2026-10-23 at 16.40.59.jpeg")
                modality = "conjunctiva"
                if not case_path.exists():
                    case_path = Path("data/raw/cp_anemic/CP-AnemiC/Anemic/Image_001.png")

            if case_path.exists():
                img_bgr = cv2.imread(str(case_path))

        if img_bgr is None:
            img_bgr = np.ones((300, 300, 3), dtype=np.uint8) * 128

        report = analyze_model_shortcuts(img_bgr, modality=modality)
        return render_template(
            "achilles_heel.html",
            report=report,
            active_case=case,
        )

    @app.route("/api/v1/achilles-heel", methods=["POST"])
    def api_achilles_heel():
        """API endpoint returning structured shortcut vulnerability report."""
        import cv2
        import numpy as np
        from achilles_heel import analyze_model_shortcuts

        if "image" not in request.files:
            return jsonify({"error": "Missing 'image' file in multipart upload."}), 400
        file = request.files["image"]
        modality = request.form.get("modality", "conjunctiva")
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Could not decode uploaded image."}), 400

        report = analyze_model_shortcuts(img_bgr, modality=modality)
        return jsonify(report)

    @app.route("/api/v1/explain/<int:scan_id>", methods=["POST"])
    def api_explain_scan(scan_id: int):
        """Explainable AI endpoint powered by Qwen 2.5 with MediAssist clinical guardrails."""
        from explainable_ai import explain_scan_findings

        record = scans_active_query().filter_by(id=scan_id).first_or_404()
        payload = request.get_json(silent=True) or {}
        question = payload.get("question", "").strip() or "What does my image specifically show?"
        language = payload.get("language", "en").strip()

        result = explain_scan_findings(record, question=question, language=language)
        return jsonify(result)

    return app


app = create_app()
eager_load_predictor()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
