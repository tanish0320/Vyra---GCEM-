"""Check every included real sample, without training or claiming model outputs."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from vyra_preprocessing import FEATURE_NAMES, extract_27_colorimetric_features, load_prepared_sample, pipeline_fingerprint
from vyra_preprocessing.artifacts import write_json
from vyra_preprocessing.datasets import verify_prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    verification = verify_prepared(args.dataset_root)
    if not verification["passed"]:
        raise SystemExit(json.dumps(verification))
    with (args.dataset_root / "manifests/sample_manifest.csv").open(newline="", encoding="utf-8") as stream:
        samples = list(csv.DictReader(stream))
    with (args.dataset_root / "manifests/features.csv").open(newline="", encoding="utf-8") as stream:
        feature_rows = {r["sample_id"]: r for r in csv.DictReader(stream)}
    included = [r for r in samples if r["split"]]
    modalities = Counter()
    for index, row in enumerate(included, 1):
        cached = load_prepared_sample(row, args.dataset_root)
        recomputed = load_prepared_sample(row, args.dataset_root, cached=False)
        if not cached.is_valid or not recomputed.is_valid:
            raise AssertionError(f"Included sample is invalid: {row['sample_id']}")
        np.testing.assert_array_equal(cached.normalized_image_rgb, recomputed.normalized_image_rgb)
        np.testing.assert_array_equal(cached.feature_vector, recomputed.feature_vector)
        np.testing.assert_array_equal(cached.feature_vector, extract_27_colorimetric_features(cached.normalized_image_rgb))
        table_vector = np.array([float(feature_rows[row["sample_id"]][name]) for name in FEATURE_NAMES], dtype=np.float32)
        np.testing.assert_array_equal(cached.feature_vector, table_vector)
        modalities[row["body_part"]] += 1
        if index % 200 == 0 or index == len(included):
            print(f"Verified raw/cache/features for {index}/{len(included)} included samples", flush=True)
    statuses = defaultdict(Counter)
    for row in samples:
        group = row["dataset"] + ":" + (row["anemia_label"] or "no_source_binary_label")
        statuses[group][row["inclusion_status"] + (":" + row["exclusion_reason"] if row["exclusion_reason"] else "")] += 1
    result = {"passed": True, "preprocessing_fingerprint": pipeline_fingerprint(),
              "source_and_manifest_verification": verification,
              "included_samples_recomputed_from_source": len(included),
              "cached_images_and_27_features_exactly_equal": True,
              "by_body_part": dict(modalities),
              "statuses_by_source_class": {k: dict(v) for k, v in statuses.items()},
              "is_model_evaluation": False}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
