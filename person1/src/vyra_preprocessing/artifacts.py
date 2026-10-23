"""Portable, strict JSON/CSV and local preview exports."""

import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
from PIL import Image

from .normalization import PreprocessingDetails


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in records:
            values = {}
            for name in fields:
                value = row.get(name)
                if value is None:
                    value = ""
                elif isinstance(value, bool):
                    value = "true" if value else "false"
                elif isinstance(value, (list, dict)):
                    value = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
                values[name] = value
            writer.writerow(values)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.system()}
    for package in ("numpy", "scipy", "opencv-python-headless", "Pillow", "openpyxl", "libarchive-c"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def result_json(details: PreprocessingDetails) -> dict:
    result = details.result
    return json_safe({
        "metadata": details.metadata,
        "feature_vector": result.feature_vector,
        "body_part_encoding": result.body_part_encoding,
        "ita_angle": result.ita_angle,
        "ita_category": result.ita_category,
        "glare_fraction": result.glare_fraction,
        "is_valid": result.is_valid,
        "quality_warnings": result.quality_warnings,
    })


def export_preview(details: PreprocessingDetails, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(details.result.raw_image_rgb).save(destination / "raw.png")
    if details.result.is_valid:
        for name, image in (("raw_roi", details.raw_roi_rgb), ("glare_corrected", details.glare_corrected_rgb),
                            ("white_balanced", details.white_balanced_rgb), ("normalized", details.result.normalized_image_rgb)):
            Image.fromarray(image).save(destination / f"{name}.png")
        Image.fromarray(details.roi_mask.astype(np.uint8) * 255).save(destination / "roi_mask.png")
    else:
        # A rejected recapture must not leave a previous patient's valid preview.
        for name in ("raw_roi", "glare_corrected", "white_balanced", "normalized", "roi_mask"):
            (destination / f"{name}.png").unlink(missing_ok=True)
    write_json(destination / "metadata.json", result_json(details))

