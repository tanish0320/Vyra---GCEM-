"""Run with --dataset-root pointing at a completed prepare output.

Uses a real manifest row; does not load or impersonate a trained model.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from vyra_preprocessing import pipeline_fingerprint, load_prepared_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()
    with (args.dataset_root / "splits" / "train.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    if row["preprocessing_fingerprint"] != pipeline_fingerprint():
        raise RuntimeError("Dataset and runtime preprocessing differ; regenerate or use the frozen environment")
    # Cached output has already been normalized; do not run preprocessing on it again.
    result = load_prepared_sample(row, args.dataset_root)
    if not result.is_valid:
        raise RuntimeError(result.quality_warnings)
    recomputed = load_prepared_sample(row, args.dataset_root, cached=False)
    np.testing.assert_array_equal(result.normalized_image_rgb, recomputed.normalized_image_rgb)
    np.testing.assert_array_equal(result.feature_vector, recomputed.feature_vector)
    print(json.dumps({"sample_id": row["sample_id"], "preprocessing_matches_cache": True,
                      "feature_shape": list(result.feature_vector.shape),
                      "body_part_encoding": result.body_part_encoding.tolist(),
                      "has_lab_hb": row["hb_available"] == "true",
                      "is_model_prediction": False}, indent=2))


if __name__ == "__main__":
    main()
