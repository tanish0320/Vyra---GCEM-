"""Command-line preparation, image previews, verification and handoff export."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

import cv2
import numpy as np
from PIL import Image

from .artifacts import export_preview, json_safe, result_json, runtime_versions
from .config import DEFAULT_CONFIG, FEATURE_SCHEMA, PREPROCESSING_VERSION, pipeline_fingerprint
from .datasets import export_handoff, prepare_datasets, segmented_eye_mask, verify_prepared
from .features import FEATURE_NAMES
from .image_io import InputError, load_image
from .normalization import preprocess_with_details


def _binary_mask(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"))
    if not np.isin(values, [0, 1, 255]).all():
        raise ValueError("Masks must be binary (0/1/255); a segmented color photograph is not a binary mask")
    return values > 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Vyra Person 1 preprocessing and data preparation")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare both real datasets from archives or directories")
    prepare.add_argument("--eyes", required=True, type=Path)
    prepare.add_argument("--nails", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--seed", default=42, type=int)
    prepare.add_argument("--phash-radius", default=6, type=int)
    image = commands.add_parser("image", help="Normalize a single image and export local previews")
    image.add_argument("image", type=Path)
    image.add_argument("--body-part", required=True, choices=["conjunctiva", "fingernail", "eye", "nail"])
    image.add_argument("--output", required=True, type=Path)
    image.add_argument("--roi-mask", type=Path)
    image.add_argument("--skin-mask", type=Path)
    image.add_argument("--bbox", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    image.add_argument("--pre-cropped", action="store_true", help="Declare a verified tissue crop")
    image.add_argument("--eda-segmented", action="store_true", help="Use the EDA-specific white-background mask")
    verify = commands.add_parser("verify", help="Recheck source/derived hashes, features and split overlap")
    verify.add_argument("--dataset-root", required=True, type=Path)
    handoff = commands.add_parser("export-handoff", help="Export CSV/JSON reports without images")
    handoff.add_argument("--dataset-root", required=True, type=Path)
    handoff.add_argument("--output", required=True, type=Path)
    commands.add_parser("fingerprint", help="Print version, feature schema, configuration and runtime")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    # Keep the CLI deterministic and avoid oversubscribing CPUs for tiny ROIs.
    # Library callers retain control of their own OpenCV thread configuration.
    cv2.setNumThreads(1)
    try:
        if args.command == "prepare":
            summary = prepare_datasets(args.eyes, args.nails, args.output, seed=args.seed,
                                       phash_radius=args.phash_radius, progress=lambda line: print(line, flush=True))
            print(json.dumps(summary, indent=2, allow_nan=False))
        elif args.command == "image":
            mask = _binary_mask(args.roi_mask)
            if args.eda_segmented:
                if mask is not None or args.bbox is not None or args.body_part not in {"conjunctiva", "eye"}:
                    raise ValueError("EDA masking applies to conjunctiva and cannot be combined with a supplied ROI")
                loaded = load_image(args.image)
                mask = loaded.alpha_mask if loaded.alpha_mask is not None else segmented_eye_mask(loaded.rgb)
            details = preprocess_with_details(args.image, args.body_part, roi_mask=mask,
                                              roi_bbox=None if args.bbox is None else tuple(args.bbox),
                                              skin_mask=_binary_mask(args.skin_mask), pre_cropped=args.pre_cropped)
            export_preview(details, args.output)
            print(json.dumps(result_json(details), indent=2, allow_nan=False))
            return 0 if details.result.is_valid else 2
        elif args.command == "verify":
            result = verify_prepared(args.dataset_root)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["passed"] else 2
        elif args.command == "export-handoff":
            export_handoff(args.dataset_root, args.output)
            print("Exported verified metadata and feature reports; no photographs included.")
        elif args.command == "fingerprint":
            print(json.dumps(json_safe({"preprocessing_version": PREPROCESSING_VERSION,
                                        "preprocessing_fingerprint": pipeline_fingerprint(),
                                        "feature_schema": FEATURE_SCHEMA, "feature_names": FEATURE_NAMES,
                                        "config": asdict(DEFAULT_CONFIG), "runtime": runtime_versions()}), indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, BadZipFile) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

