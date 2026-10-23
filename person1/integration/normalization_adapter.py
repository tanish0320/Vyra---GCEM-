"""Copy to Person 2's src/data/normalization_adapter.py after installing the package.

Final training and inference must use the same package version and fingerprint.
Opaque input requires a supplied ROI or pre_cropped=True for a verified crop.
"""

from vyra_preprocessing import (
    FEATURE_NAMES, PREPROCESSING_VERSION, PreprocessingResult, apply_hsv_clahe,
    compute_ita, encode_body_part, extract_27_colorimetric_features,
    grey_world_white_balance, pipeline_fingerprint, preprocess_image,
    preprocess_with_details, remove_specular_glare, load_prepared_sample,
)

__all__ = [
    "PreprocessingResult", "preprocess_image", "preprocess_with_details",
    "extract_27_colorimetric_features", "encode_body_part", "grey_world_white_balance",
    "remove_specular_glare", "apply_hsv_clahe", "compute_ita", "FEATURE_NAMES",
    "PREPROCESSING_VERSION", "pipeline_fingerprint",
    "load_prepared_sample",
]

