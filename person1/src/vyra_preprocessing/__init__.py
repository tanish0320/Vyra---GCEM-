"""Person 1's shared preprocessing interface."""

from .config import DEFAULT_CONFIG, FEATURE_SCHEMA, PREPROCESSING_VERSION, PipelineConfig, pipeline_fingerprint
from .features import FEATURE_COUNT, FEATURE_NAMES, extract_27_colorimetric_features
from .normalization import (
    PreprocessingDetails, PreprocessingResult, apply_hsv_clahe, compute_ita,
    encode_body_part, grey_world_white_balance, preprocess_image,
    preprocess_with_details, remove_specular_glare,
)
from .prepared import load_prepared_sample

__version__ = PREPROCESSING_VERSION
__all__ = [
    "DEFAULT_CONFIG", "FEATURE_SCHEMA", "PREPROCESSING_VERSION", "PipelineConfig",
    "pipeline_fingerprint", "FEATURE_COUNT", "FEATURE_NAMES", "PreprocessingResult",
    "PreprocessingDetails", "preprocess_image", "preprocess_with_details",
    "extract_27_colorimetric_features", "encode_body_part", "grey_world_white_balance",
    "remove_specular_glare", "apply_hsv_clahe", "compute_ita",
    "load_prepared_sample",
]

