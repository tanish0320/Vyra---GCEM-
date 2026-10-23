"""Backend integration adapter connecting VYRA Flask routes to Person 1 & Person 2 contracts.

This module implements:
- Pre-resize raw upload validation and separation of raw vs. normalized/model-input images.
- Optional integration boundary for Person 1's optical normalization pipeline.
- Integration with Person 2's exact inference callable (`src.inference.predictor.predict`).
- Strict validation and mapping of Person 2's 13-field response contract.
- Conservative, configurable provisional triage logic (RED / YELLOW / GREEN).
- Git LFS pointer detection and model readiness inspection.
- Opt-in development mock safeguards (disabled by default, blocked in production).
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageOps

from config import BEST_CHECKPOINT_PATH, ENABLE_GRADCAM, FLASK_CONFIG, TRAINING_CONFIG
from image_validator import ImageValidationResult, build_quality_payload, validate_image
from model import inspect_checkpoint_file, latest_available_checkpoint
from predict import (
    AnemiaPredictor,
    PredictionResult,
    compute_risk_metadata,
    medical_advice_for_prediction,
    medical_advice_lines,
    normalize_prediction_label,
)


logger = logging.getLogger("vyra.service_adapter")

SUPPORTED_BODY_PARTS: tuple[str, ...] = ("conjunctiva", "fingernail")
DEFAULT_SCREENING_THRESHOLD: float = 0.50
HIGH_SENSITIVITY_THRESHOLD: float = 0.255
EXPECTED_FEATURE_VECTOR_DIM: int = 27
MIN_PLAUSIBLE_HB_G_DL: float = 1.0
MAX_PLAUSIBLE_HB_G_DL: float = 25.0

VALID_ITA_CATEGORIES: frozenset[str] = frozenset(
    {"Very Light", "Light", "Intermediate", "Tan", "Brown", "Dark"}
)
VALID_CALIBRATION_STATUSES: frozenset[str] = frozenset({"calibrated", "uncalibrated"})
VALID_REGRESSION_STATUSES: frozenset[str] = frozenset({"validated", "unavailable"})

UNAVAILABLE_HB_MESSAGE: str = (
    "Continuous Hb estimate unavailable for this modality. "
    "Screening based on calibrated risk classification."
)

PROTOTYPE_MEDICAL_DISCLAIMER: str = (
    "This tool is an AI-assisted screening prototype and does not constitute a "
    "confirmatory medical diagnosis. All flagged cases require laboratory Complete "
    "Blood Count (CBC) verification."
)

_BODY_PART_ALIASES: dict[str, str] = {
    "conjunctiva": "conjunctiva",
    "eye conjunctiva": "conjunctiva",
    "eye_conjunctiva": "conjunctiva",
    "palpebral conjunctiva": "conjunctiva",
    "eye": "conjunctiva",
    "fingernail": "fingernail",
    "finger nail": "fingernail",
    "fingernails": "fingernail",
    "nail": "fingernail",
    "nailbed": "fingernail",
    "fingernail bed": "fingernail",
}

_BODY_PART_UI_LABELS: dict[str, str] = {
    "conjunctiva": "Eye Conjunctiva",
    "fingernail": "Fingernail",
}

_SEX_CODE_ALIASES: dict[str, str] = {
    "female": "F",
    "f": "F",
    "woman": "F",
    "male": "M",
    "m": "M",
    "man": "M",
    "other": "O",
    "o": "O",
    "non-binary": "O",
    "nonbinary": "O",
}

_INJECTED_PREDICTOR_FN: Callable[..., dict[str, Any]] | None = None
_INJECTED_NORMALIZER_FN: Callable[..., dict[str, Any]] | None = None


class ServiceAdapterError(Exception):
    """Base structured exception for integration adapter failures."""

    def __init__(
        self,
        user_message: str,
        *,
        status_code: int = 500,
        failure_category: str = "adapter_error",
        internal_detail: str | None = None,
    ) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.status_code = status_code
        self.failure_category = failure_category
        self.internal_detail = internal_detail or user_message


class ImageUploadValidationError(ServiceAdapterError):
    """Raised when an uploaded image fails extension, size, decoding, or quality checks."""

    def __init__(self, user_message: str, *, internal_detail: str | None = None) -> None:
        super().__init__(
            user_message,
            status_code=400,
            failure_category="invalid_upload",
            internal_detail=internal_detail,
        )


class ModalityValidationError(ServiceAdapterError):
    """Raised when the requested body_part is missing or unsupported."""

    def __init__(self, user_message: str, *, internal_detail: str | None = None) -> None:
        super().__init__(
            user_message,
            status_code=400,
            failure_category="unsupported_modality",
            internal_detail=internal_detail,
        )


class ModelUnavailableError(ServiceAdapterError):
    """Raised when real inference cannot run because the model or weights are unavailable."""

    def __init__(
        self,
        user_message: str,
        *,
        failure_category: str = "model_unavailable",
        internal_detail: str | None = None,
    ) -> None:
        super().__init__(
            user_message,
            status_code=503,
            failure_category=failure_category,
            internal_detail=internal_detail,
        )


class PredictionExecutionError(ServiceAdapterError):
    """Raised when the underlying predictor throws an unexpected runtime error or fatal error field."""

    def __init__(self, user_message: str, *, internal_detail: str | None = None) -> None:
        super().__init__(
            user_message,
            status_code=502,
            failure_category="prediction_execution_failed",
            internal_detail=internal_detail,
        )


class ContractValidationError(ServiceAdapterError):
    """Raised when the predictor returns a malformed or unusable response payload."""

    def __init__(self, user_message: str, *, internal_detail: str | None = None) -> None:
        super().__init__(
            user_message,
            status_code=502,
            failure_category="invalid_model_output",
            internal_detail=internal_detail,
        )


@dataclass(frozen=True)
class TriageConfig:
    """Configurable provisional triage thresholds per Person 2 specification."""

    red_hb_threshold_g_dl: float = 7.0
    yellow_hb_threshold_g_dl: float = 11.0
    red_probability_threshold: float = 0.85
    yellow_probability_threshold: float = 0.50
    high_sensitivity_threshold: float = HIGH_SENSITIVITY_THRESHOLD
    require_validated_hb_for_green: bool = True
    provisional_rule_status: str = "provisional_pending_clinical_signoff"


DEFAULT_TRIAGE_CONFIG = TriageConfig()


@dataclass(frozen=True)
class NormalizationStageOutput:
    """Result of the preprocessing/normalization boundary before model resize."""

    raw_image_path: Path
    normalized_image_path: Path | None
    model_input_path: Path
    feature_vector: list[float] | None
    ita_angle: float | None
    ita_category: str | None
    glare_fraction: float | None
    normalization_available: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AdapterPredictionOutput:
    """Unified output from service_adapter consumed by Flask routes, DB, and PDF reports."""

    success: bool
    is_mock: bool
    hb_estimate_g_dl: float | None
    hb_regression_available: bool
    hemoglobin_estimate: str
    anemia_probability: float | None
    anemic_probability: float
    non_anemic_probability: float
    predicted_class: str | None
    prediction: str | None
    confidence: float
    confidence_percent: float
    risk_level: str | None
    risk_explanation: str
    medical_advice: str
    medical_advice_list: list[str]
    model_version: str
    calibration_status: str
    regression_status: str
    anatomical_modality: str
    image_type_label: str
    ita_angle: float | None
    ita_category: str | None
    glare_fraction: float | None
    processing_time_ms: int | None
    quality_warnings: list[str]
    error: str | None
    triage: dict[str, Any] | None
    screening_threshold: float
    high_sensitivity_mode: bool
    raw_image_path: Path
    normalized_image_path: Path | None
    model_input_path: Path
    gradcam_path: str | None
    quality: dict[str, Any]
    disclaimer: str = PROTOTYPE_MEDICAL_DISCLAIMER

    def to_legacy_prediction_result(self) -> PredictionResult:
        """Convert into legacy PredictionResult dataclass for backward-compatible callers."""

        return PredictionResult(
            prediction=self.prediction,
            confidence=self.confidence,
            anemic_probability=self.anemic_probability,
            non_anemic_probability=self.non_anemic_probability,
            risk_level=self.risk_level,
            risk_explanation=self.risk_explanation,
            gradcam_path=self.gradcam_path,
            medical_advice=self.medical_advice,
            hemoglobin_estimate=self.hemoglobin_estimate,
            processing_time_ms=self.processing_time_ms or 0,
            image_quality=self.quality.get("image_quality"),
            error=self.error,
            warnings=list(self.quality_warnings),
        )

    def to_api_dict(self) -> dict[str, Any]:
        """Return the standardized API dictionary combining Person 2 & Flask fields."""

        return {
            "success": self.success,
            "is_mock": self.is_mock,
            "hb_estimate_g_dl": self.hb_estimate_g_dl,
            "hb_regression_available": self.hb_regression_available,
            "hemoglobin_estimate": self.hemoglobin_estimate,
            "anemia_probability": self.anemia_probability,
            "anemic_probability": self.anemic_probability,
            "non_anemic_probability": self.non_anemic_probability,
            "predicted_class": self.predicted_class,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "confidence_percent": self.confidence_percent,
            "risk_level": self.risk_level,
            "risk_explanation": self.risk_explanation,
            "medical_advice": self.medical_advice,
            "model_version": self.model_version,
            "calibration_status": self.calibration_status,
            "regression_status": self.regression_status,
            "anatomical_modality": self.anatomical_modality,
            "ita_angle": self.ita_angle,
            "ita_category": self.ita_category,
            "glare_fraction": self.glare_fraction,
            "processing_time_ms": self.processing_time_ms,
            "quality_warnings": list(self.quality_warnings),
            "warnings": list(self.quality_warnings),
            "error": self.error,
            "triage": self.triage,
            "screening_threshold": self.screening_threshold,
            "high_sensitivity_mode": self.high_sensitivity_mode,
            "disclaimer": self.disclaimer,
        }


def set_test_predictor(predictor_fn: Callable[..., dict[str, Any]] | None) -> None:
    """Register or clear a controlled test double for `src.inference.predictor.predict`."""

    global _INJECTED_PREDICTOR_FN
    _INJECTED_PREDICTOR_FN = predictor_fn


def set_test_normalizer(normalizer_fn: Callable[..., dict[str, Any]] | None) -> None:
    """Register or clear a controlled test double for Person 1's normalization pipeline."""

    global _INJECTED_NORMALIZER_FN
    _INJECTED_NORMALIZER_FN = normalizer_fn


def is_production_environment() -> bool:
    """Return True if the application is running in a production environment."""

    for env_key in ("FLASK_ENV", "ENV", "VYRA_ENV", "APP_ENV"):
        val = (os.getenv(env_key) or "").strip().lower()
        if val in {"prod", "production"}:
            return True

    if os.getenv("RENDER") or os.getenv("VERCEL") or os.getenv("DYNO"):
        return True

    return False


def is_mock_mode_enabled() -> bool:
    """Return True only when development mock mode is explicitly enabled and not in production."""

    raw_flag = (os.getenv("VYRA_ENABLE_DEV_MOCK") or "").strip().lower()
    wants_mock = raw_flag in {"1", "true", "yes"}
    if not wants_mock:
        return False

    if is_production_environment():
        logger.warning(
            "VYRA_ENABLE_DEV_MOCK was requested, but mock mode is strictly blocked in production."
        )
        return False

    return True


def resolve_body_part(raw_modality: str | None) -> str:
    """Map UI or API modality labels to Person 2's canonical body_part tokens."""

    if raw_modality is None:
        raise ModalityValidationError(
            "Anatomical modality (body_part) is required. Please select 'conjunctiva' or 'fingernail'."
        )

    cleaned = str(raw_modality).strip().lower()
    canonical = _BODY_PART_ALIASES.get(cleaned)
    if canonical not in SUPPORTED_BODY_PARTS:
        raise ModalityValidationError(
            f"Unsupported anatomical modality '{raw_modality}'. Supported modalities are: "
            "Eye Conjunctiva ('conjunctiva') and Fingernail ('fingernail')."
        )
    return canonical


def ui_label_for_body_part(body_part: str) -> str:
    """Return the human-readable UI label for a canonical body_part."""

    return _BODY_PART_UI_LABELS.get(body_part, "Unknown")


def resolve_screening_threshold(
    screening_threshold: float | str | None = None,
    high_sensitivity_mode: bool | str | None = None,
) -> tuple[float, bool]:
    """Resolve the active screening threshold and whether high-sensitivity mode is active."""

    is_high_sens = False
    if isinstance(high_sensitivity_mode, bool):
        is_high_sens = high_sensitivity_mode
    elif isinstance(high_sensitivity_mode, str):
        is_high_sens = high_sensitivity_mode.strip().lower() in {"1", "true", "yes", "on"}

    if screening_threshold not in (None, ""):
        try:
            threshold_val = float(screening_threshold)
        except (TypeError, ValueError) as exc:
            raise ServiceAdapterError(
                "screening_threshold must be a valid number between 0.05 and 0.95.",
                status_code=400,
                failure_category="invalid_threshold",
            ) from exc
        if not math.isfinite(threshold_val) or not (0.05 <= threshold_val <= 0.95):
            raise ServiceAdapterError(
                "screening_threshold must be between 0.05 and 0.95.",
                status_code=400,
                failure_category="invalid_threshold",
            )
        return round(threshold_val, 4), abs(threshold_val - HIGH_SENSITIVITY_THRESHOLD) < 1e-3

    if is_high_sens:
        return HIGH_SENSITIVITY_THRESHOLD, True

    return DEFAULT_SCREENING_THRESHOLD, False


def build_inference_metadata(patient_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Map patient intake fields to Person 2's metadata dict without leaking PII."""

    if not patient_payload:
        return {}

    metadata: dict[str, Any] = {}
    age_val = patient_payload.get("age")
    if isinstance(age_val, int) and 0 <= age_val <= 120:
        metadata["age"] = age_val
    elif age_val not in (None, ""):
        try:
            parsed_age = int(str(age_val).strip())
            if 0 <= parsed_age <= 120:
                metadata["age"] = parsed_age
        except (TypeError, ValueError):
            pass

    raw_sex = str(
        patient_payload.get("sex") or patient_payload.get("gender") or ""
    ).strip().lower()
    sex_code = _SEX_CODE_ALIASES.get(raw_sex)
    if sex_code:
        metadata["sex"] = sex_code

    return metadata


def resolve_person2_predict_callable() -> Callable[..., dict[str, Any]] | None:
    """Resolve `src.inference.predictor.predict` lazily without crashing on missing modules."""

    if _INJECTED_PREDICTOR_FN is not None:
        return _INJECTED_PREDICTOR_FN

    try:
        repo_root = Path(__file__).resolve().parent.parent
        if (repo_root / "src").is_dir() and str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        module = importlib.import_module("src.inference.predictor")
        predict_fn = getattr(module, "predict", None)
        if callable(predict_fn):
            return predict_fn
    except Exception as exc:
        logger.debug("src.inference.predictor not importable: %s", exc)

    return None


def resolve_person1_normalizer_callable() -> Callable[..., Any] | None:
    """Resolve Person 1's normalization callable if installed in the repository."""

    if _INJECTED_NORMALIZER_FN is not None:
        return _INJECTED_NORMALIZER_FN

    for module_name in (
        "vyra_preprocessing",
        "vyra_preprocessing.pipeline",
        "normalization",
        "src.preprocessing.normalization",
        "src.cv.normalization",
    ):
        try:
            module = importlib.import_module(module_name)
            for attr_name in ("preprocess_image", "normalize_image", "preprocess_and_normalize", "run_normalization"):
                candidate = getattr(module, attr_name, None)
                if callable(candidate):
                    return candidate
        except Exception:
            continue
    return None


def inspect_runtime_readiness(
    *,
    predictor_loaded: bool = False,
    predictor_load_error: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect checkpoint validity, Person 2 module availability, and overall readiness."""

    ckpt_info = inspect_checkpoint_file(checkpoint_path)
    person2_callable = resolve_person2_predict_callable()
    person1_callable = resolve_person1_normalizer_callable()

    has_injected_test_double = _INJECTED_PREDICTOR_FN is not None
    model_ready = bool(predictor_loaded or has_injected_test_double)

    failure_category = ckpt_info["failure_category"]
    reason = ckpt_info["reason"]

    if model_ready:
        failure_category = None
        reason = None
    elif ckpt_info["checkpoint_valid"] and predictor_load_error:
        failure_category = "model_load_failed"
        reason = f"Checkpoint file exists, but model initialization failed: {predictor_load_error}"
    elif ckpt_info["checkpoint_valid"] and not predictor_loaded and person2_callable is None:
        # Checkpoint is valid on disk and can be loaded on demand.
        failure_category = None
        reason = "Valid checkpoint detected; model is ready to load."

    if model_ready:
        status_message = "The AI model is loaded and ready for screening."
    elif failure_category == "lfs_pointer_checkpoint":
        status_message = (
            "Model weights are unavailable: the checkpoint file on disk is a Git LFS pointer text file "
            "rather than trained PyTorch weights."
        )
    elif failure_category == "missing_checkpoint":
        status_message = "No trained model checkpoint was found on this server."
    elif failure_category in {"corrupt_checkpoint", "unreadable_checkpoint", "model_load_failed"}:
        status_message = reason or "The model checkpoint could not be loaded."
    else:
        status_message = "The AI model is initializing."

    return {
        "checkpoint_path": ckpt_info["path"],
        "checkpoint_exists": ckpt_info["checkpoint_exists"],
        "checkpoint_is_lfs_pointer": ckpt_info["checkpoint_is_lfs_pointer"],
        "checkpoint_valid": ckpt_info["checkpoint_valid"],
        "checkpoint_size_bytes": ckpt_info["size_bytes"],
        # `checkpoint_available` is True ONLY when a valid, non-LFS checkpoint exists
        # so the UI never mistakes a 134-byte Git LFS pointer for loadable weights.
        "checkpoint_available": bool(ckpt_info["checkpoint_valid"] or has_injected_test_double),
        "person2_predictor_available": person2_callable is not None,
        "person1_normalizer_available": person1_callable is not None,
        "predictor_loaded": model_ready,
        "model_ready": model_ready,
        "failure_category": failure_category,
        "predictor_load_error": reason if failure_category else predictor_load_error,
        "status_message": status_message,
        "mock_mode_enabled": is_mock_mode_enabled(),
    }


def save_and_validate_raw_image(
    image_source: Image.Image | bytes | str | Path,
    destination: Path,
) -> tuple[Path, ImageValidationResult]:
    """Save the original image without resizing/cropping and validate its original dimensions.

    Order of operations (Phase 3 Section 4):
    1. Check file extension.
    2. Decode the original image and apply lossless EXIF orientation transpose.
    3. Save the uncropped, unresized original image to `destination`.
    4. Run `validate_image(destination)` on the original dimensions and pixels.
    5. Delete the file and raise `ImageUploadValidationError` if validation fails.
    """

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    extension = destination.suffix.lower()

    try:
        if isinstance(image_source, Image.Image):
            original = ImageOps.exif_transpose(image_source).convert("RGB")
        elif isinstance(image_source, (bytes, bytearray)):
            from io import BytesIO

            with Image.open(BytesIO(image_source)) as opened:
                original = ImageOps.exif_transpose(opened).convert("RGB")
        else:
            with Image.open(Path(image_source)) as opened:
                original = ImageOps.exif_transpose(opened).convert("RGB")

        save_kwargs: dict[str, Any]
        save_format: str
        if extension == ".png":
            save_format = "PNG"
            save_kwargs = {"optimize": True}
        elif extension == ".webp":
            save_format = "WEBP"
            save_kwargs = {"quality": 92, "method": 6}
        else:
            save_format = "JPEG"
            save_kwargs = {"quality": 95, "optimize": True}

        original.save(destination, format=save_format, **save_kwargs)
    except Exception as exc:
        if destination.exists():
            try:
                destination.unlink()
            except OSError:
                pass
        raise ImageUploadValidationError(
            "The uploaded image could not be decoded. Please upload a valid JPG, PNG, or WEBP file.",
            internal_detail=str(exc),
        ) from exc

    validation = validate_image(destination)
    if not validation.is_valid:
        if destination.exists():
            try:
                destination.unlink()
            except OSError:
                pass
        raise ImageUploadValidationError(
            validation.error or "The uploaded image did not pass validation.",
            internal_detail=f"Validation failed ({validation.width}x{validation.height}): {validation.error}",
        )

    return destination, validation


def create_separate_model_input_image(
    source_image_path: Path,
    *,
    target_size: tuple[int, int] | None = None,
) -> Path:
    """Create a separate model-input copy without overwriting the raw uploaded file."""

    source_image_path = source_image_path.resolve()
    stem = source_image_path.stem
    suffix = source_image_path.suffix or ".png"
    model_input_path = source_image_path.with_name(f"{stem}_model_input{suffix}")

    if target_size is None:
        shutil.copyfile(source_image_path, model_input_path)
        return model_input_path

    with Image.open(source_image_path) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        resized = rgb.resize(target_size, resample=Image.Resampling.LANCZOS)
        if suffix.lower() == ".png":
            resized.save(model_input_path, format="PNG")
        elif suffix.lower() == ".webp":
            resized.save(model_input_path, format="WEBP", quality=92)
        else:
            resized.save(model_input_path, format="JPEG", quality=95)
    return model_input_path


def _validate_feature_vector(
    vec: Any,
    *,
    feature_names: Any = None,
) -> list[float] | None:
    """Validate Person 1's 27-dimensional feature vector before passing to Person 2."""

    if vec is None:
        return None
    if feature_names is not None:
        if not isinstance(feature_names, (list, tuple)) or len(feature_names) != EXPECTED_FEATURE_VECTOR_DIM:
            logger.warning(
                "Ignoring feature_vector because feature_names length is invalid: %r",
                feature_names,
            )
            return None
        cleaned_names = [str(name).strip() for name in feature_names if isinstance(name, str) and str(name).strip()]
        if len(cleaned_names) != EXPECTED_FEATURE_VECTOR_DIM or len(set(cleaned_names)) != EXPECTED_FEATURE_VECTOR_DIM:
            logger.warning("Ignoring feature_vector because feature_names contain empty or duplicate entries.")
            return None

    if isinstance(vec, np.ndarray):
        if vec.ndim != 1 or vec.shape[0] != EXPECTED_FEATURE_VECTOR_DIM:
            logger.warning("Ignoring feature_vector with unexpected ndarray shape: %s", vec.shape)
            return None
        items = vec.tolist()
    elif isinstance(vec, (list, tuple)):
        if len(vec) != EXPECTED_FEATURE_VECTOR_DIM:
            logger.warning("Ignoring feature_vector with length %d (expected %d)", len(vec), EXPECTED_FEATURE_VECTOR_DIM)
            return None
        items = list(vec)
    else:
        return None

    validated: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        val = float(item)
        if not math.isfinite(val):
            return None
        validated.append(val)
    return validated


def run_normalization_stage(
    raw_image_path: Path,
    body_part: str,
) -> NormalizationStageOutput:
    """Execute Person 1's normalization pipeline if available, or use the documented boundary."""

    raw_image_path = raw_image_path.resolve()
    normalizer_fn = resolve_person1_normalizer_callable()

    if normalizer_fn is None:
        # Temporary integration boundary: keep original untouched and create a separate
        # model-input file without destructive cropping or fabricated ITA metrics.
        model_input_path = create_separate_model_input_image(raw_image_path, target_size=None)
        return NormalizationStageOutput(
            raw_image_path=raw_image_path,
            normalized_image_path=None,
            model_input_path=model_input_path,
            feature_vector=None,
            ita_angle=None,
            ita_category=None,
            glare_fraction=None,
            normalization_available=False,
            warnings=[],
        )

    normalized_preview_path = raw_image_path.with_name(
        f"{raw_image_path.stem}_normalized{raw_image_path.suffix or '.png'}"
    )
    warnings: list[str] = []

    try:
        raw_out = normalizer_fn(image=str(raw_image_path), body_part=body_part)
    except TypeError:
        raw_out = normalizer_fn(str(raw_image_path), body_part)
    except Exception as exc:
        logger.error("Person 1 normalization failed for %s: %s", raw_image_path.name, exc)
        warnings.append("Optical normalization was unavailable for this image; using uncropped original.")
        model_input_path = create_separate_model_input_image(raw_image_path, target_size=None)
        return NormalizationStageOutput(
            raw_image_path=raw_image_path,
            normalized_image_path=None,
            model_input_path=model_input_path,
            feature_vector=None,
            ita_angle=None,
            ita_category=None,
            glare_fraction=None,
            normalization_available=False,
            warnings=warnings,
        )

    norm_dict: dict[str, Any] = {}
    if isinstance(raw_out, dict):
        norm_dict = raw_out
    elif hasattr(raw_out, "to_dict") and callable(raw_out.to_dict):
        norm_dict = raw_out.to_dict()

    if norm_dict.get("is_valid") is False or norm_dict.get("error"):
        err_msg = str(norm_dict.get("error") or "Image failed optical normalization quality checks.")
        raise ImageUploadValidationError(err_msg)

    norm_img = norm_dict.get("normalized_image")
    if isinstance(norm_img, Image.Image):
        norm_img.convert("RGB").save(normalized_preview_path)
    elif isinstance(norm_img, np.ndarray) and norm_img.ndim == 3:
        Image.fromarray(norm_img.astype(np.uint8)).save(normalized_preview_path)
    elif isinstance(norm_img, (str, Path)) and Path(norm_img).exists():
        if Path(norm_img).resolve() != raw_image_path:
            shutil.copyfile(Path(norm_img), normalized_preview_path)

    effective_norm_path = normalized_preview_path if normalized_preview_path.exists() else None
    model_input_path = effective_norm_path or create_separate_model_input_image(raw_image_path, target_size=None)

    raw_fv = norm_dict.get("feature_vector")
    feature_vector = _validate_feature_vector(
        raw_fv,
        feature_names=norm_dict.get("feature_names"),
    )
    if raw_fv is not None and feature_vector is None:
        warnings.append(
            "Optical feature vector had missing/NaN values or unconfirmed 27-feature ordering; omitted from predictor call."
        )
    ita_angle = _clean_optional_float(norm_dict.get("ita_angle"), min_val=-90.0, max_val=90.0)
    ita_cat_raw = norm_dict.get("ita_category")
    ita_category = str(ita_cat_raw).strip() if isinstance(ita_cat_raw, str) and ita_cat_raw.strip() in VALID_ITA_CATEGORIES else None
    glare_fraction = _clean_optional_float(norm_dict.get("glare_fraction"), min_val=0.0, max_val=1.0)

    raw_warnings = norm_dict.get("quality_warnings") or norm_dict.get("warnings") or []
    if isinstance(raw_warnings, (list, tuple)):
        for item in raw_warnings:
            if isinstance(item, str) and item.strip():
                warnings.append(item.strip())

    return NormalizationStageOutput(
        raw_image_path=raw_image_path,
        normalized_image_path=effective_norm_path,
        model_input_path=model_input_path,
        feature_vector=feature_vector,
        ita_angle=ita_angle,
        ita_category=ita_category,
        glare_fraction=glare_fraction,
        normalization_available=True,
        warnings=warnings,
    )


def _clean_optional_float(
    value: Any,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float | None:
    """Parse an optional finite float within [min_val, max_val], returning None if invalid."""

    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    if min_val is not None and parsed < min_val:
        return None
    if max_val is not None and parsed > max_val:
        return None
    return parsed


def compute_vyra_triage(
    *,
    hb_estimate_g_dl: float | None,
    regression_status: str | None,
    anemia_probability: float | None,
    predicted_class: str | None = None,
    screening_threshold: float = DEFAULT_SCREENING_THRESHOLD,
    config: TriageConfig = DEFAULT_TRIAGE_CONFIG,
) -> dict[str, Any] | None:
    """Compute provisional CHW triage level conservatively.

    Rules (Person 2 Integration Spec + Phase 3 Section 7):
    - Only use `hb_estimate_g_dl` when `regression_status == "validated"` and `hb_estimate_g_dl` is valid.
    - RED (Urgent Referral): `validated_hb < 7.0` OR `anemia_probability > 0.85`.
    - YELLOW (PHC Referral): `7.0 <= validated_hb < 11.0` OR `anemia_probability >= effective_threshold`
      OR `predicted_class == "anemic"`.
    - GREEN (Routine Follow-up): `validated_hb >= 11.0` AND `anemia_probability < effective_threshold`
      AND `predicted_class != "anemic"`.
    - Missing / conflicting inputs:
      - If inputs conflict (e.g. `validated_hb >= 11.0` while `anemia_probability >= 0.50`),
        resolve conservatively to the higher-priority referral tier (RED or YELLOW), never GREEN.
      - Never assign GREEN when required inputs (`validated_hb` or `anemia_probability`) are missing.
    """

    validated_hb: float | None = None
    if regression_status == "validated" and hb_estimate_g_dl is not None:
        validated_hb = _clean_optional_float(
            hb_estimate_g_dl,
            min_val=MIN_PLAUSIBLE_HB_G_DL,
            max_val=MAX_PLAUSIBLE_HB_G_DL,
        )

    valid_prob = _clean_optional_float(anemia_probability, min_val=0.0, max_val=1.0)
    norm_class = (predicted_class or "").strip().lower() or None

    effective_yellow_prob = min(config.yellow_probability_threshold, screening_threshold)

    conflict_detected = False
    if validated_hb is not None and valid_prob is not None:
        if (validated_hb >= config.yellow_hb_threshold_g_dl and valid_prob >= effective_yellow_prob) or (
            validated_hb < config.yellow_hb_threshold_g_dl and valid_prob < effective_yellow_prob
        ):
            conflict_detected = True

    # 1. RED (Urgent Referral)
    red_by_hb = validated_hb is not None and validated_hb < config.red_hb_threshold_g_dl
    red_by_prob = valid_prob is not None and valid_prob > config.red_probability_threshold
    if red_by_hb or red_by_prob:
        reasons: list[str] = []
        if red_by_hb:
            reasons.append(f"Validated Hb estimate ({validated_hb:.2f} g/dL) is below {config.red_hb_threshold_g_dl:.1f} g/dL.")
        if red_by_prob:
            reasons.append(
                f"Calibrated anemia probability ({valid_prob * 100:.1f}%) exceeds {config.red_probability_threshold * 100:.0f}%."
            )
        return {
            "level": "RED",
            "label": "RED (Urgent Referral)",
            "action": "Urgent referral to FRU / Hospital for immediate laboratory CBC and clinical evaluation.",
            "reasons": reasons,
            "conflict_detected": conflict_detected,
            "provisional": True,
            "rule_status": config.provisional_rule_status,
            "disclaimer": PROTOTYPE_MEDICAL_DISCLAIMER,
        }

    # 2. YELLOW (PHC Referral)
    yellow_by_hb = (
        validated_hb is not None
        and config.red_hb_threshold_g_dl <= validated_hb < config.yellow_hb_threshold_g_dl
    )
    yellow_by_prob = valid_prob is not None and valid_prob >= effective_yellow_prob
    yellow_by_class = norm_class == "anemic"
    if yellow_by_hb or yellow_by_prob or yellow_by_class:
        reasons = []
        if yellow_by_hb:
            reasons.append(
                f"Validated Hb estimate ({validated_hb:.2f} g/dL) is between "
                f"{config.red_hb_threshold_g_dl:.1f} and {config.yellow_hb_threshold_g_dl:.1f} g/dL."
            )
        if yellow_by_prob:
            reasons.append(
                f"Anemia probability ({valid_prob * 100:.1f}%) meets or exceeds screening threshold ({effective_yellow_prob * 100:.1f}%)."
            )
        if yellow_by_class and not yellow_by_prob:
            reasons.append("Model predicted class is anemic.")
        if conflict_detected:
            reasons.append("Conflicting Hb and probability signals resolved conservatively to PHC referral.")
        return {
            "level": "YELLOW",
            "label": "YELLOW (PHC Referral)",
            "action": "Refer to Primary Health Centre (PHC) for confirmatory laboratory CBC testing.",
            "reasons": reasons,
            "conflict_detected": conflict_detected,
            "provisional": True,
            "rule_status": config.provisional_rule_status,
            "disclaimer": PROTOTYPE_MEDICAL_DISCLAIMER,
        }

    # 3. GREEN (Routine Follow-up) — requires BOTH validated Hb >= 11.0 AND valid_prob < threshold
    if config.require_validated_hb_for_green:
        if validated_hb is None or valid_prob is None:
            # Never assign GREEN when required inputs (validated Hb or probability) are missing.
            return None
        if validated_hb >= config.yellow_hb_threshold_g_dl and valid_prob < effective_yellow_prob and norm_class != "anemic":
            return {
                "level": "GREEN",
                "label": "GREEN (Routine Follow-up)",
                "action": "Routine community follow-up and nutritional counseling; order CBC if clinical symptoms exist.",
                "reasons": [
                    f"Validated Hb estimate ({validated_hb:.2f} g/dL) is at least {config.yellow_hb_threshold_g_dl:.1f} g/dL "
                    f"and anemia probability ({valid_prob * 100:.1f}%) is below {effective_yellow_prob * 100:.1f}%."
                ],
                "conflict_detected": False,
                "provisional": True,
                "rule_status": config.provisional_rule_status,
                "disclaimer": PROTOTYPE_MEDICAL_DISCLAIMER,
            }
        return None

    if valid_prob is not None and valid_prob < effective_yellow_prob and norm_class != "anemic":
        return {
            "level": "GREEN",
            "label": "GREEN (Routine Follow-up)",
            "action": "Routine follow-up; confirm with CBC if clinical symptoms are present.",
            "reasons": [f"Anemia probability ({valid_prob * 100:.1f}%) is below {effective_yellow_prob * 100:.1f}%."],
            "conflict_detected": False,
            "provisional": True,
            "rule_status": config.provisional_rule_status,
            "disclaimer": PROTOTYPE_MEDICAL_DISCLAIMER,
        }

    return None


def normalize_person2_response(
    raw_response: dict[str, Any],
    *,
    body_part: str,
    screening_threshold: float,
    high_sensitivity_mode: bool,
    raw_image_path: Path,
    normalized_stage: NormalizationStageOutput,
    validation: ImageValidationResult,
    is_mock: bool = False,
    gradcam_path: str | None = None,
    triage_config: TriageConfig = DEFAULT_TRIAGE_CONFIG,
) -> AdapterPredictionOutput:
    """Validate and map Person 2's response contract into AdapterPredictionOutput."""

    if not isinstance(raw_response, dict):
        raise ContractValidationError(
            "The AI inference service returned an invalid response structure.",
            internal_detail=f"Expected dict from predict(), got {type(raw_response).__name__}",
        )

    fatal_error = raw_response.get("error")
    if isinstance(fatal_error, str) and fatal_error.strip():
        raise PredictionExecutionError(
            fatal_error.strip(),
            internal_detail=f"Predictor reported fatal error: {fatal_error.strip()}",
        )
    if fatal_error not in (None, "", False):
        raise PredictionExecutionError(
            "The AI model reported a fatal error while processing this scan.",
            internal_detail=f"Non-null error field: {fatal_error!r}",
        )

    warnings: list[str] = []
    for w in validation.warnings:
        if isinstance(w, str) and w.strip() and w.strip() not in warnings:
            warnings.append(w.strip())
    for w in normalized_stage.warnings:
        if isinstance(w, str) and w.strip() and w.strip() not in warnings:
            warnings.append(w.strip())

    raw_warnings = raw_response.get("quality_warnings")
    if isinstance(raw_warnings, (list, tuple)):
        for item in raw_warnings:
            if isinstance(item, str) and item.strip() and item.strip() not in warnings:
                warnings.append(item.strip())
    elif raw_warnings not in (None, []):
        warnings.append("Model returned non-standard quality_warnings field.")

    # 1. Regression status & Hb estimate rule
    raw_reg_status = raw_response.get("regression_status")
    if isinstance(raw_reg_status, str) and raw_reg_status.strip().lower() in VALID_REGRESSION_STATUSES:
        regression_status = raw_reg_status.strip().lower()
    else:
        regression_status = "unavailable"
        if raw_reg_status not in (None, ""):
            warnings.append("Unrecognized regression_status from model; treated as unavailable.")

    raw_hb = raw_response.get("hb_estimate_g_dl")
    hb_estimate_g_dl: float | None = None
    if raw_hb is not None:
        parsed_hb = _clean_optional_float(
            raw_hb,
            min_val=MIN_PLAUSIBLE_HB_G_DL,
            max_val=MAX_PLAUSIBLE_HB_G_DL,
        )
        if parsed_hb is None:
            warnings.append("Model returned an invalid or out-of-range hb_estimate_g_dl; value was suppressed.")
        elif regression_status == "validated":
            hb_estimate_g_dl = round(parsed_hb, 2)
        else:
            # Rule: suppress numeric Hb estimate unless regression_status == "validated"
            hb_estimate_g_dl = None
            warnings.append(
                "Numeric Hb estimate was suppressed because regression_status is not 'validated'."
            )

    hb_regression_available = bool(hb_estimate_g_dl is not None and regression_status == "validated")
    if hb_regression_available and hb_estimate_g_dl is not None:
        hemoglobin_estimate_display = f"{hb_estimate_g_dl:.2f} g/dL"
    else:
        hemoglobin_estimate_display = UNAVAILABLE_HB_MESSAGE

    # 2. Anemia probability validation (must be in [0.0, 1.0])
    raw_prob = raw_response.get("anemia_probability")
    anemia_probability: float | None = None
    if raw_prob is not None:
        parsed_prob = _clean_optional_float(raw_prob, min_val=0.0, max_val=1.0)
        if parsed_prob is None:
            warnings.append(
                "Model returned an invalid or out-of-range anemia_probability; probability was discarded."
            )
        else:
            anemia_probability = round(parsed_prob, 4)

    # 3. Predicted class validation
    raw_class = raw_response.get("predicted_class")
    predicted_class: str | None = None
    if isinstance(raw_class, str) and raw_class.strip():
        token = raw_class.strip().lower().replace("_", "-")
        if token in {"anemic", "non-anemic"}:
            predicted_class = token
        else:
            warnings.append(f"Unrecognized predicted_class '{raw_class}' from model; ignored.")
    elif raw_class is not None:
        warnings.append("Malformed predicted_class type from model; ignored.")

    if predicted_class is None and anemia_probability is not None:
        predicted_class = "anemic" if anemia_probability >= screening_threshold else "non-anemic"

    if predicted_class is None and anemia_probability is None:
        raise ContractValidationError(
            "The AI model returned neither a valid anemia_probability nor a valid predicted_class.",
            internal_detail=f"Invalid contract payload: {raw_response!r}",
        )

    prediction_ui_label = normalize_prediction_label(predicted_class) if predicted_class else None

    if anemia_probability is not None:
        anemic_prob_float = anemia_probability
        non_anemic_prob_float = round(1.0 - anemia_probability, 4)
        confidence_float = (
            anemic_prob_float if predicted_class == "anemic" else non_anemic_prob_float
        )
    else:
        anemic_prob_float = 0.0
        non_anemic_prob_float = 0.0
        confidence_float = 0.0

    # 4. Calibration status (never label uncalibrated output as calibrated)
    raw_cal = raw_response.get("calibration_status")
    if isinstance(raw_cal, str) and raw_cal.strip().lower() in VALID_CALIBRATION_STATUSES:
        calibration_status = raw_cal.strip().lower()
    else:
        calibration_status = "uncalibrated"

    # 5. Model version
    raw_version = raw_response.get("model_version")
    model_version = (
        str(raw_version).strip()
        if isinstance(raw_version, str) and raw_version.strip()
        else "unknown"
    )

    # 6. Modality
    raw_modality = raw_response.get("anatomical_modality")
    if isinstance(raw_modality, str) and raw_modality.strip().lower() in SUPPORTED_BODY_PARTS:
        anatomical_modality = raw_modality.strip().lower()
    else:
        anatomical_modality = body_part

    # 7. ITA & Glare fields (never invent values)
    ita_angle = _clean_optional_float(raw_response.get("ita_angle"), min_val=-90.0, max_val=90.0)
    if ita_angle is None:
        ita_angle = normalized_stage.ita_angle
    else:
        ita_angle = round(ita_angle, 2)

    raw_ita_cat = raw_response.get("ita_category")
    if isinstance(raw_ita_cat, str) and raw_ita_cat.strip() in VALID_ITA_CATEGORIES:
        ita_category: str | None = raw_ita_cat.strip()
    else:
        ita_category = normalized_stage.ita_category

    glare_fraction = _clean_optional_float(raw_response.get("glare_fraction"), min_val=0.0, max_val=1.0)
    if glare_fraction is None:
        glare_fraction = normalized_stage.glare_fraction
    else:
        glare_fraction = round(glare_fraction, 4)

    # 8. Processing time
    raw_time = raw_response.get("processing_time_ms")
    processing_time_ms: int | None = None
    if isinstance(raw_time, (int, float)) and not isinstance(raw_time, bool) and math.isfinite(raw_time) and raw_time >= 0:
        processing_time_ms = int(round(raw_time))

    # 9. Risk metadata & conservative triage
    if prediction_ui_label and anemia_probability is not None:
        risk_level, risk_explanation = compute_risk_metadata(prediction_ui_label, confidence_float)
        advice_list = medical_advice_lines(prediction_ui_label)
        advice_text = medical_advice_for_prediction(prediction_ui_label)
    else:
        risk_level = "Uncertain"
        risk_explanation = (
            "Calibrated probability was unavailable; clinical review and laboratory CBC testing are recommended."
        )
        advice_list = [
            "Arrange a laboratory Complete Blood Count (CBC) test to verify hemoglobin status.",
            PROTOTYPE_MEDICAL_DISCLAIMER,
        ]
        advice_text = " ".join(advice_list)

    if calibration_status != "calibrated":
        risk_explanation = f"{risk_explanation} (Note: probability output is {calibration_status}.)"

    triage = compute_vyra_triage(
        hb_estimate_g_dl=hb_estimate_g_dl,
        regression_status=regression_status,
        anemia_probability=anemia_probability,
        predicted_class=predicted_class,
        screening_threshold=screening_threshold,
        config=triage_config,
    )

    if is_mock:
        mock_banner = (
            "DEMO MOCK MODE: Simulated output for development testing only. "
            "Not a real clinical or AI prediction."
        )
        if mock_banner not in warnings:
            warnings.insert(0, mock_banner)

    quality_payload = build_quality_payload(validation)
    quality_payload["warnings"] = list(warnings)
    quality_payload["warning"] = warnings[0] if warnings else None

    return AdapterPredictionOutput(
        success=True,
        is_mock=is_mock,
        hb_estimate_g_dl=hb_estimate_g_dl,
        hb_regression_available=hb_regression_available,
        hemoglobin_estimate=hemoglobin_estimate_display,
        anemia_probability=anemia_probability,
        anemic_probability=anemic_prob_float,
        non_anemic_probability=non_anemic_prob_float,
        predicted_class=predicted_class,
        prediction=prediction_ui_label,
        confidence=confidence_float,
        confidence_percent=round(confidence_float * 100.0, 1),
        risk_level=risk_level,
        risk_explanation=risk_explanation,
        medical_advice=advice_text,
        medical_advice_list=advice_list,
        model_version=model_version,
        calibration_status=calibration_status,
        regression_status=regression_status,
        anatomical_modality=anatomical_modality,
        image_type_label=ui_label_for_body_part(anatomical_modality),
        ita_angle=ita_angle,
        ita_category=ita_category,
        glare_fraction=glare_fraction,
        processing_time_ms=processing_time_ms,
        quality_warnings=warnings,
        error=None,
        triage=triage,
        screening_threshold=screening_threshold,
        high_sensitivity_mode=high_sensitivity_mode,
        raw_image_path=raw_image_path,
        normalized_image_path=normalized_stage.normalized_image_path,
        model_input_path=normalized_stage.model_input_path,
        gradcam_path=gradcam_path,
        quality=quality_payload,
        disclaimer=PROTOTYPE_MEDICAL_DISCLAIMER,
    )


def _run_legacy_predictor_bridge(
    predictor: AnemiaPredictor,
    *,
    model_input_path: Path,
    body_part: str,
    metadata: dict[str, Any],
    screening_threshold: float,
) -> tuple[dict[str, Any], str | None]:
    """Adapt an initialized legacy AnemiaPredictor to Person 2's response schema.

    Because the legacy baseline model has only a binary classification head,
    `hb_estimate_g_dl` is always `None` and `regression_status` is `"unavailable"`.
    """

    legacy_result = predictor.predict_image(
        image_path=model_input_path,
        patient_info=metadata,
        save_gradcam=ENABLE_GRADCAM,
    )
    if legacy_result.error:
        return {"error": legacy_result.error, "quality_warnings": legacy_result.warnings}, None

    anemia_prob = float(legacy_result.anemic_probability)
    predicted_class = "anemic" if anemia_prob >= screening_threshold else "non-anemic"
    temp_value = getattr(getattr(predictor, "temperature_scaler", None), "temperature", 1.0)
    is_calibrated = abs(float(temp_value) - 1.0) > 1e-6

    arch = "efficientnet_b3"
    if isinstance(getattr(predictor, "checkpoint", None), dict):
        arch = str(predictor.checkpoint.get("architecture", arch))

    contract_dict: dict[str, Any] = {
        "hb_estimate_g_dl": None,
        "anemia_probability": round(anemia_prob, 4),
        "predicted_class": predicted_class,
        "model_version": f"legacy-{arch}-binary",
        "calibration_status": "calibrated" if is_calibrated else "uncalibrated",
        "quality_warnings": list(legacy_result.warnings),
        "error": None,
        "anatomical_modality": body_part,
        "ita_angle": None,
        "ita_category": None,
        "glare_fraction": None,
        "processing_time_ms": legacy_result.processing_time_ms,
        "regression_status": "unavailable",
    }
    return contract_dict, legacy_result.gradcam_path


def _build_dev_mock_response(
    *,
    body_part: str,
    screening_threshold: float,
    processing_time_ms: int,
) -> dict[str, Any]:
    """Return an explicitly labeled development mock response (never used in production)."""

    mock_prob = 0.62
    return {
        "hb_estimate_g_dl": None,
        "anemia_probability": mock_prob,
        "predicted_class": "anemic" if mock_prob >= screening_threshold else "non-anemic",
        "model_version": "vyra-dev-mock-not-clinical",
        "calibration_status": "uncalibrated",
        "quality_warnings": [
            "DEMO MOCK MODE: Simulated output for development testing only. Not a real clinical or AI prediction."
        ],
        "error": None,
        "anatomical_modality": body_part,
        "ita_angle": None,
        "ita_category": None,
        "glare_fraction": None,
        "processing_time_ms": processing_time_ms,
        "regression_status": "unavailable",
    }


def run_vyra_predict(
    *,
    raw_image_path: str | Path,
    body_part: str | None,
    patient_payload: dict[str, Any] | None = None,
    screening_threshold: float | str | None = None,
    high_sensitivity_mode: bool | str | None = None,
    feature_vector: list[float] | np.ndarray | None = None,
    predictor_fn: Callable[..., dict[str, Any]] | None = None,
    fallback_predictor: AnemiaPredictor | None = None,
    use_mock: bool = False,
    triage_config: TriageConfig = DEFAULT_TRIAGE_CONFIG,
) -> AdapterPredictionOutput:
    """Execute the end-to-end Phase 3 prediction pipeline through Person 2's contract."""

    started_at = perf_counter()
    resolved_raw_path = Path(raw_image_path).expanduser().resolve()

    # 1. Validate raw uploaded image BEFORE any normalization or model resizing
    validation = validate_image(resolved_raw_path)
    if not validation.is_valid:
        raise ImageUploadValidationError(
            validation.error or "The uploaded image failed quality validation."
        )

    # 2. Validate & canonicalize body_part ("conjunctiva" or "fingernail")
    canonical_body_part = resolve_body_part(body_part)

    # 3. Resolve screening threshold (0.50 balanced default, 0.255 high-sensitivity)
    resolved_threshold, is_high_sens = resolve_screening_threshold(
        screening_threshold=screening_threshold,
        high_sensitivity_mode=high_sensitivity_mode,
    )

    # 4. Build sanitized patient metadata dictionary ({"age": ..., "sex": ...})
    metadata = build_inference_metadata(patient_payload)

    # 5. Run Person 1 normalization stage (or documented boundary preserving raw image)
    normalized_stage = run_normalization_stage(resolved_raw_path, canonical_body_part)
    validated_feature_vector = (
        _validate_feature_vector(feature_vector)
        if feature_vector is not None
        else normalized_stage.feature_vector
    )

    # 6. Check explicit development mock mode
    if use_mock:
        if not is_mock_mode_enabled():
            raise ModelUnavailableError(
                "Development mock mode is disabled or blocked in this environment.",
                failure_category="mock_mode_disabled",
            )
        elapsed_ms = max(1, int(round((perf_counter() - started_at) * 1000)))
        mock_raw = _build_dev_mock_response(
            body_part=canonical_body_part,
            screening_threshold=resolved_threshold,
            processing_time_ms=elapsed_ms,
        )
        return normalize_person2_response(
            mock_raw,
            body_part=canonical_body_part,
            screening_threshold=resolved_threshold,
            high_sensitivity_mode=is_high_sens,
            raw_image_path=resolved_raw_path,
            normalized_stage=normalized_stage,
            validation=validation,
            is_mock=True,
            gradcam_path=None,
            triage_config=triage_config,
        )

    # 7. Resolve Person 2's `src.inference.predictor.predict` callable
    active_predict_fn = predictor_fn or resolve_person2_predict_callable()
    gradcam_path: str | None = None

    if active_predict_fn is not None:
        # If using the real `src.inference.predictor.predict` (not an injected test double),
        # verify that checkpoint files are not Git LFS pointers if no test double is injected.
        if predictor_fn is None and _INJECTED_PREDICTOR_FN is None:
            ckpt_check = inspect_checkpoint_file()
            if ckpt_check["checkpoint_is_lfs_pointer"]:
                raise ModelUnavailableError(
                    "Model weights are unavailable because the checkpoint file on disk is a Git LFS pointer file.",
                    failure_category="lfs_pointer_checkpoint",
                    internal_detail=ckpt_check["reason"],
                )
        try:
            raw_result = active_predict_fn(
                image=str(normalized_stage.model_input_path),
                feature_vector=validated_feature_vector,
                body_part=canonical_body_part,
                metadata=metadata,
                screening_threshold=resolved_threshold,
            )
        except ServiceAdapterError:
            raise
        except Exception as exc:
            logger.exception("Person 2 predict() execution failed: %s", exc)
            raise PredictionExecutionError(
                "The AI inference engine encountered an error while analyzing this image.",
                internal_detail=str(exc),
            ) from exc

        if isinstance(raw_result, dict) and raw_result.get("gradcam_path"):
            candidate_gradcam = str(raw_result["gradcam_path"]).strip()
            candidate_path = Path(candidate_gradcam)
            gradcam_dir = Path(FLASK_CONFIG.gradcam_folder)
            gradcam_dir.mkdir(parents=True, exist_ok=True)
            if candidate_path.is_file():
                dest_gradcam = gradcam_dir / candidate_path.name
                if candidate_path.resolve() != dest_gradcam.resolve():
                    shutil.copyfile(candidate_path, dest_gradcam)
                gradcam_path = f"gradcam/{dest_gradcam.name}"
            elif (gradcam_dir / candidate_path.name).is_file():
                gradcam_path = f"gradcam/{candidate_path.name}"
            else:
                gradcam_path = None
        elif gradcam_path is None and ENABLE_GRADCAM and isinstance(raw_result, dict) and not raw_result.get("error"):
            try:
                model_to_use = None
                import src.inference.predictor as pred_module
                if getattr(pred_module, "_GLOBAL_ENGINE", None) and pred_module._GLOBAL_ENGINE.model is not None:
                    model_to_use = pred_module._GLOBAL_ENGINE.model
                elif fallback_predictor is not None:
                    model_to_use = fallback_predictor.model

                if model_to_use is not None:
                    from gradcam import generate_and_save_gradcam
                    import cv2
                    img_bgr = cv2.imread(str(normalized_stage.model_input_path))
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        img_norm = (img_resized - mean) / std
                        dev = next(model_to_use.parameters()).device
                        inp_tensor = torch.tensor(img_norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(dev)
                        pred_class_idx = 1 if (raw_result.get("predicted_class") == "anemic" or (raw_result.get("anemia_probability") or 0.0) >= resolved_threshold) else 0
                        gradcam_dir = Path(FLASK_CONFIG.gradcam_folder)
                        gradcam_dir.mkdir(parents=True, exist_ok=True)
                        artifacts = generate_and_save_gradcam(
                            model=model_to_use,
                            image_path=normalized_stage.model_input_path,
                            input_tensor=inp_tensor,
                            predicted_class_index=pred_class_idx,
                            output_dir=gradcam_dir,
                        )
                        gradcam_path = f"gradcam/{artifacts.output_path.name}"
            except Exception as exc:
                logger.debug("Automatic Grad-CAM generation skipped: %s", exc)
    elif fallback_predictor is not None:
        try:
            raw_result, gradcam_path = _run_legacy_predictor_bridge(
                fallback_predictor,
                model_input_path=normalized_stage.model_input_path,
                body_part=canonical_body_part,
                metadata=metadata,
                screening_threshold=resolved_threshold,
            )
        except Exception as exc:
            logger.exception("Fallback AnemiaPredictor execution failed: %s", exc)
            raise PredictionExecutionError(
                "The AI inference engine encountered an error while analyzing this image.",
                internal_detail=str(exc),
            ) from exc
    else:
        readiness = inspect_runtime_readiness(predictor_loaded=False)
        raise ModelUnavailableError(
            readiness["status_message"],
            failure_category=readiness["failure_category"] or "model_unavailable",
            internal_detail=readiness.get("predictor_load_error"),
        )

    if isinstance(raw_result, dict) and raw_result.get("processing_time_ms") is None:
        raw_result = dict(raw_result)
        raw_result["processing_time_ms"] = max(1, int(round((perf_counter() - started_at) * 1000)))

    return normalize_person2_response(
        raw_result,
        body_part=canonical_body_part,
        screening_threshold=resolved_threshold,
        high_sensitivity_mode=is_high_sens,
        raw_image_path=resolved_raw_path,
        normalized_stage=normalized_stage,
        validation=validation,
        is_mock=False,
        gradcam_path=gradcam_path,
        triage_config=triage_config,
    )
