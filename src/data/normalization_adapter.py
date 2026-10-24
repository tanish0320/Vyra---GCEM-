"""Optical Normalization & Feature Extraction Adapter (Person 1 Interface).

Provides:
- Grey-World White Balance
- Specular Glare Masking
- HSV-CLAHE Illumination Normalization
- CIE LAB Individual Typology Angle (ITA) Calculation
- 27-dimension Camera-Invariant Colorimetric Feature Vector
- Body-Part Encoding (Conjunctiva vs Fingernail)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


@dataclass
class PreprocessingResult:
    """Output contract for Person 1 preprocessing module."""
    raw_image_rgb: np.ndarray
    normalized_image_rgb: np.ndarray
    feature_vector: np.ndarray  # 27 colorimetric features
    body_part_encoding: np.ndarray  # [is_conjunctiva, is_fingernail]
    ita_angle: float
    ita_category: str
    glare_fraction: float
    is_valid: bool
    quality_warnings: List[str] = field(default_factory=list)


def grey_world_white_balance(image_rgb: np.ndarray, max_gain_deviation: float = 0.15) -> np.ndarray:
    """Apply Grey-World white balance assumption to neutralize illumination color casts.
    
    Channel gains are clamped to [1 - max_gain_deviation, 1 + max_gain_deviation] to prevent
    destructive bleaching of biological tissue erythema (pink nail beds and conjunctiva)
    into achromatic gray.
    """
    img_float = image_rgb.astype(np.float32)
    mean_r = np.mean(img_float[:, :, 0]) + 1e-6
    mean_g = np.mean(img_float[:, :, 1]) + 1e-6
    mean_b = np.mean(img_float[:, :, 2]) + 1e-6
    mean_gray = (mean_r + mean_g + mean_b) / 3.0

    min_g = max(0.5, 1.0 - max_gain_deviation)
    max_g = min(1.8, 1.0 + max_gain_deviation)
    gain_r = float(np.clip(mean_gray / mean_r, min_g, max_g))
    gain_g = float(np.clip(mean_gray / mean_g, min_g, max_g))
    gain_b = float(np.clip(mean_gray / mean_b, min_g, max_g))

    img_float[:, :, 0] = np.clip(img_float[:, :, 0] * gain_r, 0, 255)
    img_float[:, :, 1] = np.clip(img_float[:, :, 1] * gain_g, 0, 255)
    img_float[:, :, 2] = np.clip(img_float[:, :, 2] * gain_b, 0, 255)

    return img_float.astype(np.uint8)


def remove_specular_glare(image_rgb: np.ndarray, sat_thresh: int = 40, val_thresh: int = 225) -> Tuple[np.ndarray, float]:
    """Detect specular glare reflections (high V, low S) and inpaint them via cv2.inpaint."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    glare_mask = ((val > val_thresh) & (sat < sat_thresh)).astype(np.uint8) * 255
    glare_fraction = float(np.mean(glare_mask > 0))

    if glare_fraction > 0.001:
        inpainted = cv2.inpaint(image_rgb, glare_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return inpainted, glare_fraction
    return image_rgb, glare_fraction


def apply_hsv_clahe(image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Standardize illumination by applying CLAHE strictly to the HSV Value channel."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def compute_ita(image_rgb: np.ndarray) -> Tuple[float, str]:
    """Calculate Individual Typology Angle (ITA) in CIE L*a*b* space:
    ITA = arctan((L* - 50) / b*) * (180 / pi)
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV maps L* to [0, 255], a* to [0, 255], b* to [0, 255]
    # Standard CIE L* is in [0, 100], b* is in [-128, 127]
    L = lab[:, :, 0] * (100.0 / 255.0)
    b = lab[:, :, 2] - 128.0

    mean_L = float(np.mean(L))
    mean_b = float(np.mean(b))
    if abs(mean_b) < 1e-4:
        mean_b = 1e-4

    ita = float(np.arctan((mean_L - 50.0) / mean_b) * (180.0 / np.pi))

    if ita > 55:
        category = "Very Light"
    elif ita > 41:
        category = "Light"
    elif ita > 28:
        category = "Intermediate"
    elif ita > 10:
        category = "Tan"
    elif ita > -30:
        category = "Brown"
    else:
        category = "Dark"

    return ita, category


def extract_27_colorimetric_features(image_rgb: np.ndarray) -> np.ndarray:
    """Extract standard camera-invariant colorimetric ratios, pallor indices, and spatial gradients.
    
    Produces 27 features:
    1. Brightness mean
    2. Brightness std
    3-7. Brightness percentiles (10, 25, 50, 75, 90)
    8. Redness ratio mean (R / (R+G+B))
    9. Redness ratio std
    10. White pallor ratio (>180)
    11. Pink ratio
    12-17. R, G, B channel means and stds (6)
    18. (R_mean + 1) / (G_mean + 1)
    19. (R_mean + 1) / (B_mean + 1)
    20. (R_mean - B_mean) / 255
    21. Hemoglobin proxy mean (R / (G+B+1))
    22. Hemoglobin proxy std
    23. Top-Bottom spatial gradient
    24. Center-Surround contrast
    25. Spatial gradient magnitude mean
    26. Spatial gradient magnitude std
    27. Local texture variance mean
    """
    img_resized = cv2.resize(image_rgb, (224, 224)).astype(np.float32)
    r = img_resized[:, :, 0]
    g = img_resized[:, :, 1]
    b = img_resized[:, :, 2]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    features = []

    # 1-2: Brightness mean, std
    features.append(float(np.mean(brightness)))
    features.append(float(np.std(brightness)))

    # 3-7: Percentiles
    for p in [10, 25, 50, 75, 90]:
        features.append(float(np.percentile(brightness, p)))

    # 8-9: Redness features
    redness = r / (r + g + b + 1e-6)
    features.append(float(np.mean(redness)))
    features.append(float(np.std(redness)))

    # 10-11: Pallor features
    white_ratio = float(np.sum(brightness > 180) / brightness.size)
    pink_ratio = float(np.sum((r > 150) & (g < 150) & (b < 150)) / brightness.size)
    features.append(white_ratio)
    features.append(pink_ratio)

    # 12-17: Channel statistics
    for ch in [r, g, b]:
        features.append(float(np.mean(ch)))
        features.append(float(np.std(ch)))

    # 18-20: Color ratios
    features.append(float((np.mean(r) + 1.0) / (np.mean(g) + 1.0)))
    features.append(float((np.mean(r) + 1.0) / (np.mean(b) + 1.0)))
    features.append(float((np.mean(r) - np.mean(b)) / 255.0))

    # 21-22: Hemoglobin proxy
    hb_proxy = r / (g + b + 1.0)
    features.append(float(np.mean(hb_proxy)))
    features.append(float(np.std(hb_proxy)))

    # 23-24: Spatial features
    h, w = brightness.shape
    top = np.mean(brightness[: h // 3, :])
    bottom = np.mean(brightness[2 * h // 3 :, :])
    features.append(float(top - bottom))

    center = np.mean(brightness[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4])
    features.append(float(center - np.mean(brightness)))

    # 25-26: Gradient features
    gx = np.abs(np.diff(brightness, axis=1, prepend=brightness[:, :1]))
    gy = np.abs(np.diff(brightness, axis=0, prepend=brightness[:1, :]))
    gradient = np.sqrt(gx**2 + gy**2)
    features.append(float(np.mean(gradient)))
    features.append(float(np.std(gradient)))

    # 27: Local texture variance
    local_mean = uniform_filter(brightness, size=7)
    local_var = uniform_filter((brightness - local_mean) ** 2, size=7)
    features.append(float(np.mean(np.sqrt(np.maximum(local_var, 0)))))

    assert len(features) == 27, f"Expected 27 features, got {len(features)}"
    return np.array(features, dtype=np.float32)


def encode_body_part(body_part: Optional[str]) -> np.ndarray:
    """Encode body part as 2D one-hot vector: [is_conjunctiva, is_fingernail]."""
    if not body_part:
        return np.array([0.5, 0.5], dtype=np.float32)
    token = body_part.strip().lower()
    if "nail" in token or "finger" in token:
        return np.array([0.0, 1.0], dtype=np.float32)
    elif "eye" in token or "conj" in token:
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([0.5, 0.5], dtype=np.float32)


def preprocess_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    body_part: Optional[str] = None,
) -> PreprocessingResult:
    """Deterministic end-to-end preprocessing pipeline aligned with Person 1's contract."""
    warnings = []

    # Load image as RGB numpy array
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            return PreprocessingResult(
                raw_image_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
                normalized_image_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
                feature_vector=np.zeros(27, dtype=np.float32),
                body_part_encoding=encode_body_part(body_part),
                ita_angle=0.0,
                ita_category="Unknown",
                glare_fraction=0.0,
                is_valid=False,
                quality_warnings=[f"Image file does not exist: {path}"],
            )
        # Prefer OpenCV for robust handling of PNG iCCP profiles and alpha channels
        mat = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mat is not None:
            if len(mat.shape) == 2:
                raw_rgb = cv2.cvtColor(mat, cv2.COLOR_GRAY2RGB)
            elif mat.shape[2] == 4:
                # RGBA / BGRA format (e.g. CP-AnemiC or EDA)
                bgr = mat[:, :, :3]
                alpha = mat[:, :, 3]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # If transparent background exists, keep RGB as is
                raw_rgb = rgb
            else:
                raw_rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        else:
            try:
                pil_img = Image.open(path).convert("RGB")
                raw_rgb = np.array(pil_img)
            except Exception as e:
                return PreprocessingResult(
                    raw_image_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
                    normalized_image_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
                    feature_vector=np.zeros(27, dtype=np.float32),
                    body_part_encoding=encode_body_part(body_part),
                    ita_angle=0.0,
                    ita_category="Unknown",
                    glare_fraction=0.0,
                    is_valid=False,
                    quality_warnings=[f"Unable to read image file: {e}"],
                )
    elif isinstance(image, Image.Image):
        raw_rgb = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        raw_rgb = image.copy()
        if len(raw_rgb.shape) == 2:
            raw_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    h, w = raw_rgb.shape[:2]
    if raw_rgb.dtype == np.uint16:
        raw_rgb = (raw_rgb / 256).astype(np.uint8)
    elif raw_rgb.dtype != np.uint8:
        raw_rgb = np.clip(raw_rgb, 0, 255).astype(np.uint8)

    if h < 50 or w < 50:
        warnings.append(f"Image dimensions ({w}x{h}) are too small for reliable screening.")

    # 1. Specular glare masking & inpainting
    deglared, glare_fraction = remove_specular_glare(raw_rgb)
    if glare_fraction > 0.15:
        warnings.append(f"Significant specular glare detected ({glare_fraction*100:.1f}%). Inpainting applied.")

    # 2. Grey-World White Balance
    wb_img = grey_world_white_balance(deglared)

    # 3. HSV-CLAHE illumination standardization
    norm_img = apply_hsv_clahe(wb_img)

    # 4. CIE LAB ITA calculation
    ita, ita_cat = compute_ita(raw_rgb)

    # 5. Extract 27 colorimetric features
    features = extract_27_colorimetric_features(norm_img)

    # 6. Encode body part
    body_enc = encode_body_part(body_part)

    return PreprocessingResult(
        raw_image_rgb=raw_rgb,
        normalized_image_rgb=norm_img,
        feature_vector=features,
        body_part_encoding=body_enc,
        ita_angle=ita,
        ita_category=ita_cat,
        glare_fraction=glare_fraction,
        is_valid=True,
        quality_warnings=warnings,
    )
