"""Helpers for portable media storage across local, Docker, and Render deployments."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal

from config import GRADCAM_DIR, STATIC_DIR, UPLOADS_DIR


StorageKind = Literal["uploads", "gradcam"]


def _storage_directories(kind: StorageKind) -> list[Path]:
    """Return the primary storage directory plus any legacy fallback locations."""

    primary = UPLOADS_DIR if kind == "uploads" else GRADCAM_DIR
    legacy = STATIC_DIR / kind
    directories = [primary.resolve()]
    if legacy.resolve() not in directories:
        directories.append(legacy.resolve())
    return directories


def extract_storage_filename(value: str | Path | None) -> str | None:
    """Extract a filename from stored Windows, POSIX, or URL-like path values."""

    if value in (None, ""):
        return None

    normalized = str(value).strip().split("?", 1)[0].split("#", 1)[0].replace("\\", "/").rstrip("/")
    if not normalized:
        return None

    filename = PurePosixPath(normalized).name
    if filename in {"", ".", ".."}:
        return None
    return filename


def storage_reference_for_path(value: str | Path | None, kind: StorageKind) -> str | None:
    """Store media as a portable '<kind>/<filename>' reference."""

    filename = extract_storage_filename(value)
    if not filename:
        return None
    return f"{kind}/{filename}"


def resolve_storage_path(value: str | Path | None, kind: StorageKind) -> Path | None:
    """Resolve a stored media reference into the best local file path."""

    if value in (None, ""):
        return None

    raw_value = str(value).strip()
    if not raw_value:
        return None

    native_path = Path(raw_value).expanduser()
    if native_path.is_absolute():
        resolved_native = native_path.resolve()
        if resolved_native.exists():
            return resolved_native

    filename = extract_storage_filename(raw_value)
    if not filename:
        return None

    candidate_directories = _storage_directories(kind)
    for directory in candidate_directories:
        candidate = (directory / filename).resolve()
        if candidate.exists():
            return candidate

    return (candidate_directories[0] / filename).resolve()
