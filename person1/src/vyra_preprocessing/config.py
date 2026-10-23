"""Versioned, deterministic preprocessing configuration."""

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path

PREPROCESSING_VERSION = "1.0.0"
FEATURE_SCHEMA = "person2-color27-v1"


@dataclass(frozen=True)
class PipelineConfig:
    min_dimension: int = 50
    glare_saturation: int = 40
    glare_value: int = 225
    glare_warning_fraction: float = 0.15
    glare_rejection_fraction: float = 0.50
    min_inpaint_fraction: float = 0.001
    inpaint_radius: float = 3.0
    clahe_clip_limit: float = 2.0
    clahe_grid: tuple[int, int] = (8, 8)
    min_skin_pixels: int = 16

    def __post_init__(self):
        object.__setattr__(self, "clahe_grid", tuple(self.clahe_grid))
        if self.min_dimension < 1 or self.min_skin_pixels < 1:
            raise ValueError("Minimum dimensions and skin-pixel count must be positive")
        if not 0 <= self.glare_saturation <= 255 or not 0 <= self.glare_value <= 255:
            raise ValueError("Glare thresholds must be between 0 and 255")
        if not 0 <= self.min_inpaint_fraction <= self.glare_warning_fraction < self.glare_rejection_fraction <= 1:
            raise ValueError("Glare fractions must be ordered between zero and one")
        if self.clahe_clip_limit <= 0 or self.inpaint_radius <= 0 or any(v < 1 for v in self.clahe_grid):
            raise ValueError("CLAHE and inpainting settings must be positive")


DEFAULT_CONFIG = PipelineConfig()


@lru_cache(maxsize=32)
def pipeline_fingerprint(config: PipelineConfig = DEFAULT_CONFIG) -> str:
    """Includes implementation bytes so edited code cannot reuse stale prepared data."""
    implementation = hashlib.sha256()
    for name in ("config.py", "features.py", "image_io.py", "normalization.py", "prepared.py"):
        implementation.update(Path(__file__).with_name(name).read_bytes())
    payload = {
        "version": PREPROCESSING_VERSION,
        "feature_schema": FEATURE_SCHEMA,
        "config": asdict(config),
        "implementation_sha256": implementation.hexdigest(),
        "runtime": {name: importlib.metadata.version(name) for name in
                    ("numpy", "scipy", "opencv-python-headless", "Pillow")},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

