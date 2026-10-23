"""Read a prepared manifest row without double-normalizing cached images."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .archives import safe_destination
from .artifacts import file_sha256
from .config import pipeline_fingerprint
from .features import FEATURE_COUNT, FEATURE_NAMES
from .image_io import load_image
from .normalization import PreprocessingResult, encode_body_part, preprocess_image


def load_prepared_sample(row: dict, dataset_root: str | Path, *, cached: bool = True) -> PreprocessingResult:
    """Uses source-coordinate mask reconstruction for uncached preprocessing.

    The cropped mask exported by preparation cannot be passed directly as a
    full-frame roi_mask. This helper performs the required coordinate mapping.
    """
    root = Path(dataset_root).resolve()
    if row.get("inclusion_status") != "included" or row.get("is_valid") not in (True, "true"):
        raise ValueError("Manifest row is not an included, valid training/evaluation sample")
    if row.get("preprocessing_fingerprint") != pipeline_fingerprint():
        raise ValueError("Prepared sample and installed preprocessing fingerprints differ")
    source = safe_destination(root, row["image_path"])
    sidecar_path = safe_destination(root, row["metadata_path"])
    for path_key, hash_key in (("image_path", "source_sha256"), ("metadata_path", "metadata_sha256"),
                               ("normalized_path", "normalized_sha256"), ("roi_mask_path", "roi_mask_sha256")):
        artifact = safe_destination(root, row[path_key])
        if not artifact.is_file() or file_sha256(artifact) != row[hash_key]:
            raise ValueError(f"Prepared sample artifact hash mismatch: {path_key}")
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if data["metadata"]["preprocessing_fingerprint"] != pipeline_fingerprint() or not data["is_valid"]:
        raise ValueError("Prepared sidecar is invalid or belongs to a different pipeline")
    if data["metadata"]["feature_names"] != list(FEATURE_NAMES) or data["metadata"].get("source_asset_id") != row["sample_id"]:
        raise ValueError("Prepared sidecar feature order or source identity is incorrect")
    raw = load_image(source).rgb
    if not cached:
        x0, y0, x1, y1 = data["metadata"]["roi_bbox"]
        h, w = raw.shape[:2]
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("Prepared ROI coordinates do not match the source")
        with Image.open(safe_destination(root, row["roi_mask_path"])) as image:
            crop_mask = np.asarray(image.convert("L")) > 0
        if crop_mask.shape != (y1 - y0, x1 - x0):
            raise ValueError("Prepared ROI mask dimensions do not match the coordinate mapping")
        mask = np.zeros(raw.shape[:2], dtype=bool)
        mask[y0:y1, x0:x1] = crop_mask
        return preprocess_image(source, row["body_part"], roi_mask=mask, pre_cropped=True)
    with Image.open(safe_destination(root, row["normalized_path"])) as image:
        normalized = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    features = np.array(data["feature_vector"], dtype=np.float32)
    body = np.array(data["body_part_encoding"], dtype=np.float32)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all() or body.shape != (2,):
        raise ValueError("Prepared feature vector or body encoding is malformed")
    if not np.array_equal(body, encode_body_part(row["body_part"])):
        raise ValueError("Prepared body encoding does not match the source modality")
    return PreprocessingResult(raw, normalized, features, body,
                               math.nan if data["ita_angle"] is None else float(data["ita_angle"]),
                               data["ita_category"], float(data["glare_fraction"]), True,
                               list(data["quality_warnings"]))

