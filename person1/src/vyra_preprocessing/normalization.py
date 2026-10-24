"""ROI-aware preprocessing; shared by training, evaluation and inference."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .config import DEFAULT_CONFIG, FEATURE_SCHEMA, PREPROCESSING_VERSION, PipelineConfig, pipeline_fingerprint
from .features import FEATURE_NAMES, extract_27_colorimetric_features
from .image_io import InputError, load_image


@dataclass
class PreprocessingResult:
    raw_image_rgb: np.ndarray
    normalized_image_rgb: np.ndarray
    feature_vector: np.ndarray
    body_part_encoding: np.ndarray
    ita_angle: float
    ita_category: str
    glare_fraction: float
    is_valid: bool
    quality_warnings: list[str] = field(default_factory=list)


@dataclass
class PreprocessingDetails:
    result: PreprocessingResult
    metadata: dict[str, Any]
    roi_mask: np.ndarray | None = None
    raw_roi_rgb: np.ndarray | None = None
    glare_corrected_rgb: np.ndarray | None = None
    white_balanced_rgb: np.ndarray | None = None


_BODY_PARTS = {
    "eye": "conjunctiva", "eyes": "conjunctiva", "conj": "conjunctiva",
    "conjunctiva": "conjunctiva", "conjunctival": "conjunctiva",
    "palpebral_conjunctiva": "conjunctiva", "nail": "fingernail",
    "nails": "fingernail", "finger": "fingernail", "fingers": "fingernail",
    "fingernail": "fingernail", "fingernails": "fingernail",
}


def encode_body_part(body_part: str | None) -> np.ndarray:
    token = _BODY_PARTS.get(body_part.strip().lower()) if isinstance(body_part, str) else None
    values = [1.0, 0.0] if token == "conjunctiva" else [0.0, 1.0] if token == "fingernail" else [0.5, 0.5]
    return np.array(values, dtype=np.float32)


def _mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.shape != shape:
        raise InputError("INVALID_MASK", f"{name} must match the EXIF-oriented image dimensions")
    if mask.dtype != bool and (mask.dtype != np.uint8 or not np.isin(mask, [0, 1, 255]).all()):
        raise InputError("INVALID_MASK", f"{name} must be boolean or uint8 with values 0, 1 or 255")
    return mask.astype(bool)


def extend_tissue(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        raise InputError("EMPTY_ROI", "No tissue pixels remain in the ROI")
    if mask.all():
        return image_rgb.copy()
    indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return np.ascontiguousarray(image_rgb[indices[0], indices[1]])


def grey_world_white_balance(image_rgb: np.ndarray, *, roi_mask: np.ndarray | None = None) -> np.ndarray:
    img_float = image_rgb.astype(np.float32)
    if roi_mask is None:
        means = [np.mean(img_float[:, :, i]) + 1e-6 for i in range(3)]
    else:
        valid = _mask(roi_mask, image_rgb.shape[:2], "roi_mask")
        if not valid.any():
            raise InputError("EMPTY_ROI", "White balance needs tissue pixels")
        means = [np.mean(img_float[:, :, i][valid]) + 1e-6 for i in range(3)]
    gray = sum(means) / 3.0
    min_g = max(0.5, 1.0 - 0.15)
    max_g = min(1.8, 1.0 + 0.15)
    for i, mean in enumerate(means):
        gain = float(np.clip(gray / mean, min_g, max_g))
        img_float[:, :, i] = np.clip(img_float[:, :, i] * gain, 0, 255)
    return img_float.astype(np.uint8)


def remove_specular_glare(image_rgb: np.ndarray, sat_thresh: int = 40, val_thresh: int = 225,
                          *, roi_mask: np.ndarray | None = None, inpaint_radius: float = 3,
                          min_inpaint_fraction: float = 0.001) -> tuple[np.ndarray, float]:
    valid = np.ones(image_rgb.shape[:2], dtype=bool) if roi_mask is None else _mask(roi_mask, image_rgb.shape[:2], "roi_mask")
    if not valid.any():
        raise InputError("EMPTY_ROI", "Glare detection needs tissue pixels")
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    glare = (hsv[:, :, 1] < sat_thresh) & (hsv[:, :, 2] > val_thresh) & valid
    fraction = float(np.count_nonzero(glare) / np.count_nonzero(valid))
    if fraction > min_inpaint_fraction:
        return cv2.inpaint(image_rgb, glare.astype(np.uint8) * 255, inpaint_radius, cv2.INPAINT_TELEA), fraction
    return image_rgb.copy(), fraction


def apply_hsv_clahe(image_rgb: np.ndarray, clip_limit: float = 2.0,
                    tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def compute_ita(image_rgb: np.ndarray, *, skin_mask: np.ndarray | None = None,
                min_skin_pixels: int = 16) -> tuple[float, str]:
    """Without a mask, this low-level helper expects an already-isolated skin patch."""
    valid = np.ones(image_rgb.shape[:2], dtype=bool) if skin_mask is None else _mask(skin_mask, image_rgb.shape[:2], "skin_mask")
    if np.count_nonzero(valid) < min_skin_pixels:
        return math.nan, "Unknown"
    lab = cv2.cvtColor(image_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    mean_l = float(np.mean(lab[:, :, 0][valid]))
    mean_b = float(np.mean(lab[:, :, 2][valid]))
    if abs(mean_b) < 1e-4:
        return math.nan, "Unknown"
    angle = float(np.degrees(np.arctan((mean_l - 50.0) / mean_b)))
    category = next((name for threshold, name in [(55, "Very Light"), (41, "Light"), (28, "Intermediate"), (10, "Tan"), (-30, "Brown")] if angle > threshold), "Dark")
    return angle, category


def _invalid(body_encoding: np.ndarray, warnings: list[str], raw: np.ndarray | None,
             metadata: dict[str, Any], glare_fraction: float = 0.0) -> PreprocessingDetails:
    placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
    result = PreprocessingResult(placeholder.copy() if raw is None else raw, placeholder,
                                np.zeros(27, dtype=np.float32), body_encoding, math.nan,
                                "Unknown", glare_fraction, False, warnings)
    metadata["is_valid"] = False
    return PreprocessingDetails(result, metadata)


def preprocess_with_details(image: str | Path | Image.Image | np.ndarray, body_part: str | None = None,
                            *, roi_mask: np.ndarray | None = None,
                            roi_bbox: tuple[int, int, int, int] | None = None,
                            skin_mask: np.ndarray | None = None, pre_cropped: bool = False,
                            config: PipelineConfig = DEFAULT_CONFIG) -> PreprocessingDetails:
    """Masks and x0,y0,x1,y1 boxes use EXIF-oriented source-image coordinates."""
    metadata: dict[str, Any] = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_fingerprint": pipeline_fingerprint(config),
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "config": asdict(config),
    }
    warnings: list[str] = []
    raw = None
    body = encode_body_part(body_part)
    glare_fraction = 0.0
    try:
        if np.array_equal(body, [0.5, 0.5]):
            raise InputError("UNSUPPORTED_BODY_PART", "Specify conjunctiva/eye or fingernail/nail")
        loaded = load_image(image)
        raw = loaded.rgb
        metadata.update(loaded.metadata)
        if loaded.metadata["png_profile_crc_repaired"]:
            warnings.append("PNG_PROFILE_CRC_REPAIRED: ancillary checksum repaired in memory; source unchanged")
        h, w = raw.shape[:2]
        if min(h, w) < config.min_dimension:
            raise InputError("IMAGE_TOO_SMALL", f"Source dimensions {w}x{h} are below {config.min_dimension}px")
        if roi_mask is not None and roi_bbox is not None:
            raise InputError("CONFLICTING_ROI", "Supply a mask or a bounding box, not both")
        if roi_mask is not None:
            tissue = _mask(roi_mask, (h, w), "roi_mask")
            metadata["roi_source"] = "explicit_mask"
        elif roi_bbox is not None:
            if len(roi_bbox) != 4 or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in roi_bbox):
                raise InputError("INVALID_ROI", "ROI box must contain four integer coordinates")
            x0, y0, x1, y1 = roi_bbox
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise InputError("INVALID_ROI", "ROI box is outside the oriented image or is empty")
            tissue = np.zeros((h, w), dtype=bool)
            tissue[y0:y1, x0:x1] = True
            metadata["roi_source"] = "explicit_bbox"
        elif loaded.alpha_mask is not None:
            tissue = loaded.alpha_mask.copy()
            metadata["roi_source"] = "alpha"
        elif pre_cropped:
            tissue = np.ones((h, w), dtype=bool)
            metadata["roi_source"] = "caller_declared_crop"
        else:
            raise InputError("ROI_REQUIRED", "Supply a tissue ROI or explicitly set pre_cropped=True for a verified crop")
        if loaded.alpha_mask is not None:
            tissue &= loaded.alpha_mask
        if not tissue.any():
            raise InputError("EMPTY_ROI", "The ROI contains no opaque tissue pixels")
        ys, xs = np.nonzero(tissue)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        mask = tissue[y0:y1, x0:x1].copy()
        raw_roi = raw[y0:y1, x0:x1].copy()
        extended = extend_tissue(raw_roi, mask)
        metadata.update({"roi_bbox": [x0, y0, x1, y1], "roi_size": [x1 - x0, y1 - y0],
                         "tissue_pixel_count": int(mask.sum()), "tissue_coverage": float(mask.mean()),
                         "background_fill": "nearest_tissue_during_processing_then_neutral_127",
                         "output_background_rgb": [127, 127, 127]})
        if min(mask.shape) < config.min_dimension:
            warnings.append("SMALL_ROI: source passed resolution gate but isolated tissue is narrower than 50px")
        deglared, glare_fraction = remove_specular_glare(
            extended, config.glare_saturation, config.glare_value, roi_mask=mask,
            inpaint_radius=config.inpaint_radius, min_inpaint_fraction=config.min_inpaint_fraction)
        metadata["glare_detected_fraction"] = glare_fraction
        metadata["glare_inpainted_fraction"] = glare_fraction if glare_fraction > config.min_inpaint_fraction else 0.0
        if glare_fraction >= config.glare_rejection_fraction:
            raise InputError("EXCESSIVE_GLARE", "At least half of the tissue is saturated; recapture the image")
        if glare_fraction > config.glare_warning_fraction:
            warnings.append("SIGNIFICANT_GLARE: more than 15% of tissue required glare handling")
        wb = grey_world_white_balance(deglared, roi_mask=mask)
        norm = apply_hsv_clahe(wb, config.clahe_clip_limit, config.clahe_grid)
        # Nearest-tissue extension supports boundary processing, but its radial
        # patterns are not observations. Never expose that invented texture to
        # the model or spatial features. Keep the recorded tissue mask alongside.
        norm[~mask] = 127
        features = extract_27_colorimetric_features(norm)
        angle, category = math.nan, "Unknown"
        if skin_mask is not None:
            skin = _mask(skin_mask, (h, w), "skin_mask")
            if np.any(skin & tissue):
                raise InputError("INVALID_SKIN_MASK", "Surrounding skin mask must not include the tissue ROI")
            angle, category = compute_ita(raw, skin_mask=skin, min_skin_pixels=config.min_skin_pixels)
        if not math.isfinite(angle):
            warnings.append("ITA_UNAVAILABLE: identified surrounding skin is absent or insufficient")
        gray = cv2.cvtColor(extended, cv2.COLOR_RGB2GRAY)
        metadata.update({"blur_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                         "raw_tissue_brightness_mean": float(gray[mask].mean()),
                         "is_valid": True, "normalized_size": [norm.shape[1], norm.shape[0]]})
        result = PreprocessingResult(raw.copy(), norm, features, body, angle, category, glare_fraction, True, warnings)
        return PreprocessingDetails(result, metadata, mask, raw_roi, deglared, wb)
    except InputError as exc:
        warnings.append(f"{exc.code}: {exc}")
        metadata["error_code"] = exc.code
        return _invalid(body, warnings, raw, metadata, glare_fraction)


def preprocess_image(image: str | Path | Image.Image | np.ndarray, body_part: str | None = None,
                     *, roi_mask: np.ndarray | None = None,
                     roi_bbox: tuple[int, int, int, int] | None = None,
                     skin_mask: np.ndarray | None = None, pre_cropped: bool = False,
                     config: PipelineConfig = DEFAULT_CONFIG) -> PreprocessingResult:
    return preprocess_with_details(image, body_part, roi_mask=roi_mask, roi_bbox=roi_bbox,
                                   skin_mask=skin_mask, pre_cropped=pre_cropped, config=config).result

