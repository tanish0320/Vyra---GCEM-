"""Person 2's exact 27-feature recipe; values are not laboratory Hb estimates."""

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

FEATURE_NAMES = (
    "brightness_mean", "brightness_std", "brightness_p10", "brightness_p25",
    "brightness_p50", "brightness_p75", "brightness_p90", "redness_mean",
    "redness_std", "white_pallor_ratio", "pink_ratio", "red_mean", "red_std",
    "green_mean", "green_std", "blue_mean", "blue_std", "red_green_ratio",
    "red_blue_ratio", "red_blue_difference", "hb_color_proxy_mean",
    "hb_color_proxy_std", "top_bottom_brightness", "center_global_brightness",
    "gradient_mean", "gradient_std", "local_std_mean",
)
FEATURE_COUNT = 27


def extract_27_colorimetric_features(image_rgb: np.ndarray) -> np.ndarray:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Features require an H x W x 3 RGB array")
    if image_rgb.dtype != np.uint8 or not all(image_rgb.shape[:2]):
        raise ValueError("Features require a nonempty uint8 RGB array")
    # Keep resize, arithmetic, population std, diff gradients and local filter
    # identical to the supplied normalization_adapter.py, not a new approximation.
    img_resized = cv2.resize(image_rgb, (224, 224)).astype(np.float32)
    r, g, b = img_resized[:, :, 0], img_resized[:, :, 1], img_resized[:, :, 2]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    features = [float(np.mean(brightness)), float(np.std(brightness))]
    for p in [10, 25, 50, 75, 90]:
        features.append(float(np.percentile(brightness, p)))
    redness = r / (r + g + b + 1e-6)
    features.extend([float(np.mean(redness)), float(np.std(redness))])
    features.append(float(np.sum(brightness > 180) / brightness.size))
    features.append(float(np.sum((r > 150) & (g < 150) & (b < 150)) / brightness.size))
    for ch in [r, g, b]:
        features.extend([float(np.mean(ch)), float(np.std(ch))])
    features.extend([
        float((np.mean(r) + 1.0) / (np.mean(g) + 1.0)),
        float((np.mean(r) + 1.0) / (np.mean(b) + 1.0)),
        float((np.mean(r) - np.mean(b)) / 255.0),
    ])
    hb_proxy = r / (g + b + 1.0)
    features.extend([float(np.mean(hb_proxy)), float(np.std(hb_proxy))])
    h, w = brightness.shape
    features.append(float(np.mean(brightness[:h // 3, :]) - np.mean(brightness[2 * h // 3:, :])))
    center = np.mean(brightness[h // 4:3 * h // 4, w // 4:3 * w // 4])
    features.append(float(center - np.mean(brightness)))
    gx = np.abs(np.diff(brightness, axis=1, prepend=brightness[:, :1]))
    gy = np.abs(np.diff(brightness, axis=0, prepend=brightness[:1, :]))
    gradient = np.sqrt(gx ** 2 + gy ** 2)
    features.extend([float(np.mean(gradient)), float(np.std(gradient))])
    local_mean = uniform_filter(brightness, size=7)
    local_var = uniform_filter((brightness - local_mean) ** 2, size=7)
    features.append(float(np.mean(np.sqrt(np.maximum(local_var, 0)))))
    result = np.array(features, dtype=np.float32)
    if result.shape != (FEATURE_COUNT,) or not np.isfinite(result).all():
        raise ValueError("Invalid feature vector")
    return result

