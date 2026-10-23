"""Dataset integrity checker for AnemiaVision AI."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_ROOT, LOGS_DIR  # noqa: E402
from dataset import CLASS_NAMES, discover_dataset_structure, list_image_files  # noqa: E402


def verify_image(image_path: Path) -> str | None:
    """Return an error message when an image is unreadable, otherwise None."""

    try:
        with Image.open(image_path) as image:
            image.verify()

        raw_buffer = np.fromfile(str(image_path), dtype=np.uint8)
        decoded = cv2.imdecode(raw_buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            return "OpenCV could not decode the image."
    except Exception as exc:  # pragma: no cover - diagnostic script
        return str(exc)

    return None


def print_summary(summary: dict[str, dict[str, int]]) -> None:
    """Print class counts per split in a compact table."""

    print("\nDataset image counts")
    print("-" * 56)
    print(f"{'Split':<10}{'Anemic':>12}{'Non-anemic':>16}{'Total':>10}")
    print("-" * 56)
    for split_name, counts in summary.items():
        total = counts["anemic"] + counts["non-anemic"]
        print(
            f"{split_name:<10}"
            f"{counts['anemic']:>12}"
            f"{counts['non-anemic']:>16}"
            f"{total:>10}"
        )
    print("-" * 56)


def save_preview_grid(
    structure: dict[str, dict[str, list[Path]]],
    output_path: Path,
    samples_per_class: int,
    show: bool,
) -> None:
    """Create a preview image grid from random examples across splits/classes."""

    random.seed(42)
    rows = len(structure) * len(CLASS_NAMES)
    cols = samples_per_class
    figure, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    row_index = 0
    for split_name, class_map in structure.items():
        for class_name in CLASS_NAMES:
            image_paths: list[Path] = []
            for class_dir in class_map[class_name]:
                image_paths.extend(list_image_files(class_dir))

            preview_paths = random.sample(image_paths, k=min(samples_per_class, len(image_paths)))
            for column in range(cols):
                axis = axes[row_index, column]
                axis.axis("off")

                if column < len(preview_paths):
                    with Image.open(preview_paths[column]) as image:
                        axis.imshow(image.convert("RGB"))
                    axis.set_title(f"{split_name} | {class_name}", fontsize=10)
            row_index += 1

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def main() -> int:
    """Run the dataset readiness checks."""

    parser = argparse.ArgumentParser(description="Validate dataset readiness for training.")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--show", action="store_true", help="Display the sample grid locally.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    structure = discover_dataset_structure(dataset_root)

    summary: dict[str, dict[str, int]] = {}
    corrupt_images: list[tuple[Path, str]] = []

    for split_name, class_map in structure.items():
        summary[split_name] = {"anemic": 0, "non-anemic": 0}

        for class_name in CLASS_NAMES:
            image_paths: list[Path] = []
            for class_dir in class_map[class_name]:
                image_paths.extend(list_image_files(class_dir))

            summary[split_name][class_name] = len(image_paths)

            for image_path in image_paths:
                error_message = verify_image(image_path)
                if error_message is not None:
                    corrupt_images.append((image_path, error_message))

    print_summary(summary)

    preview_path = LOGS_DIR / "dataset_preview.png"
    save_preview_grid(
        structure=structure,
        output_path=preview_path,
        samples_per_class=args.samples_per_class,
        show=args.show,
    )
    print(f"Saved dataset preview grid to: {preview_path}")

    if corrupt_images:
        print("\nCorrupt or unreadable images detected")
        print("-" * 56)
        for image_path, error_message in corrupt_images[:25]:
            print(f"{image_path}: {error_message}")
        if len(corrupt_images) > 25:
            print(f"... and {len(corrupt_images) - 25} more")
        print("\nDataset is NOT ready for training until corrupt files are fixed.")
        return 1

    print("\nDataset is ready for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
