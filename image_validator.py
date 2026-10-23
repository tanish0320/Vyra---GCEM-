"""Image validation utilities for the AnemiaVision AI prediction pipeline.

The hospital-facing web app and the standalone prediction helpers both rely on
this module so that uploads are checked consistently before inference starts.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import UPLOAD_EXTENSIONS


SUPPORTED_UPLOAD_EXTENSIONS = set(UPLOAD_EXTENSIONS)
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100


def _read_float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


BLUR_THRESHOLD = _read_float_env("ANEMIA_BLUR_THRESHOLD", 100.0)
LOW_LIGHT_THRESHOLD = _read_float_env("ANEMIA_LOW_LIGHT_THRESHOLD", 40.0)
HIGH_LIGHT_THRESHOLD = _read_float_env("ANEMIA_HIGH_LIGHT_THRESHOLD", 240.0)


@dataclass(frozen=True)
class ImageValidationResult:
    """Structured response returned by the validator."""

    is_valid: bool
    image_quality: str | None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    blur_score: float = 0.0
    brightness_mean: float = 0.0
    file_extension: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass into a JSON-friendly payload."""

        return asdict(self)


def is_supported_upload(filename: str | Path) -> bool:
    """Return whether a path or filename uses a supported prediction format."""

    return Path(filename).suffix.lower() in SUPPORTED_UPLOAD_EXTENSIONS


def _read_image_bgr(image_path: Path) -> np.ndarray | None:
    """Load an image from disk with Unicode-safe Windows support."""

    try:
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError:
        return None

    if encoded.size == 0:
        return None

    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def validate_image(image_path: str | Path) -> ImageValidationResult:
    """Validate one uploaded scan image before inference.

    Rejection rules:
    - Unsupported file extension
    - Unreadable image
    - Smaller than 100x100 pixels

    Warning rules:
    - Too blurry based on Laplacian variance
    - Too dark based on mean pixel intensity
    - Overexposed images are allowed, but flagged for the caller
    """

    path = Path(image_path).expanduser().resolve()
    extension = path.suffix.lower()

    if not path.exists():
        return ImageValidationResult(
            is_valid=False,
            image_quality=None,
            error="The selected image could not be found. Please upload the scan again.",
            file_extension=extension,
        )

    if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_UPLOAD_EXTENSIONS))
        return ImageValidationResult(
            is_valid=False,
            image_quality=None,
            error=f"Unsupported file format. Please use one of: {allowed}.",
            file_extension=extension,
        )

    image_bgr = _read_image_bgr(path)
    if image_bgr is None:
        return ImageValidationResult(
            is_valid=False,
            image_quality=None,
            error="The uploaded image could not be read. Please choose a different image file.",
            file_extension=extension,
        )

    height, width = image_bgr.shape[:2]
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    brightness_mean = float(np.mean(grayscale))

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        return ImageValidationResult(
            is_valid=False,
            image_quality=None,
            error=(
                "Image resolution is too small. Please upload an image that is at least "
                f"{MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT} pixels."
            ),
            width=width,
            height=height,
            blur_score=blur_score,
            brightness_mean=brightness_mean,
            file_extension=extension,
        )

    warnings: list[str] = []
    image_quality = "Good"

    if brightness_mean < LOW_LIGHT_THRESHOLD:
        image_quality = "Poor lighting"
        message = (
            "The image is quite dark. The AI can still provide a prediction, but reliability may be lower. Retake it in brighter, even lighting if possible."
        )
        warnings.append(message)

    if blur_score < BLUR_THRESHOLD:
        if image_quality == "Good":
            image_quality = "Blurry"
        message = (
            "The image appears blurry. The AI can still provide a prediction, but reliability may be lower. Retake it with a steadier hand and better focus if possible."
        )
        warnings.append(message)

    if brightness_mean > HIGH_LIGHT_THRESHOLD:
        warnings.append(
            "The image appears overexposed. The AI can still provide a prediction, but a retake may improve reliability."
        )

    return ImageValidationResult(
        is_valid=True,
        image_quality=image_quality,
        warnings=warnings,
        width=width,
        height=height,
        blur_score=blur_score,
        brightness_mean=brightness_mean,
        file_extension=extension,
    )


def build_quality_payload(result: ImageValidationResult) -> dict[str, Any]:
    """Convert validator output into a UI-friendly quality summary."""

    label_map = {
        "Good": "Good quality",
        "Blurry": "Blurry image",
        "Poor lighting": "Poor lighting",
    }
    is_good = result.is_valid and (result.image_quality == "Good")
    icon = "check" if is_good else "warning"

    if result.image_quality in label_map:
        label = label_map[result.image_quality]
    elif result.error:
        label = result.error
    else:
        label = "Unable to assess"

    return {
        "label": label,
        "icon": icon,
        "is_good": is_good,
        "score": round(result.blur_score, 2),
        "brightness": round(result.brightness_mean, 2),
        "warning": result.warnings[0] if result.warnings else None,
        "warnings": list(result.warnings),
        "error": result.error,
        "image_quality": result.image_quality,
    }
