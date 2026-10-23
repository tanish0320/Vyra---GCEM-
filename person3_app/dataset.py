"""Dataset discovery and DataLoader utilities for AnemiaVision AI."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import (
    AUGMENTATION_CONFIG,
    CLASS_NAMES,
    CLASS_TO_IDX,
    DATASET_ROOT,
    IDX_TO_CLASS,
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TRAINING_CONFIG,
)


SPLIT_NAMES = ("train", "valid", "test")


def normalize_token(value: str) -> str:
    """Normalize text for case-insensitive and punctuation-insensitive matching."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())


def canonical_split_name(folder_name: str) -> str:
    """Map a split folder name to the canonical train/valid/test form."""

    token = normalize_token(folder_name)
    split_aliases = {
        "train": "train",
        "training": "train",
        "valid": "valid",
        "validation": "valid",
        "val": "valid",
        "test": "test",
        "testing": "test",
    }
    if token not in split_aliases:
        raise ValueError(f"Unsupported dataset split folder: '{folder_name}'.")
    return split_aliases[token]


def canonical_class_name(folder_name: str) -> str:
    """Map a class folder name to the canonical anemic/non-anemic label."""

    token = normalize_token(folder_name)

    if token.startswith("non") and "anem" in token:
        return "non-anemic"

    if token in {"anemia", "anemic"} or ("anem" in token and not token.startswith("non")):
        return "anemic"

    raise ValueError(f"Unsupported class folder: '{folder_name}'.")


@dataclass(frozen=True)
class ImageSample:
    """One dataset example with its metadata."""

    path: Path
    label: int
    class_name: str
    split_name: str
    source_directory: Path


@dataclass(frozen=True)
class SplitStatistics:
    """Class counts and imbalance details for a single split."""

    split_name: str
    class_counts: dict[str, int]
    total_images: int
    imbalance_ratio: float


@dataclass(frozen=True)
class DataLoaderBundle:
    """Container returned by create_dataloaders."""

    train_dataset: "AnemiaImageDataset"
    valid_dataset: "AnemiaImageDataset"
    test_dataset: "AnemiaImageDataset"
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader
    statistics: dict[str, SplitStatistics]
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]


class AnemiaImageDataset(Dataset):
    """Simple dataset wrapper around a discovered image list."""

    def __init__(
        self,
        samples: list[ImageSample],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.targets = [sample.label for sample in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as image:
                image = image.convert("RGB")
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            raise RuntimeError(f"Failed to read image: {sample.path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.label


def discover_dataset_structure(dataset_root: Path = DATASET_ROOT) -> dict[str, dict[str, list[Path]]]:
    """Discover train/valid/test splits and their class folders automatically."""

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    structure: dict[str, dict[str, list[Path]]] = {
        split: defaultdict(list) for split in SPLIT_NAMES
    }

    for split_dir in dataset_root.iterdir():
        if not split_dir.is_dir():
            continue

        try:
            split_name = canonical_split_name(split_dir.name)
        except ValueError:
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            try:
                class_name = canonical_class_name(class_dir.name)
            except ValueError:
                continue

            structure[split_name][class_name].append(class_dir.resolve())

    missing_splits = [split for split in SPLIT_NAMES if not structure[split]]
    if missing_splits:
        raise FileNotFoundError(
            "Missing expected dataset split(s): " + ", ".join(missing_splits)
        )

    for split_name in SPLIT_NAMES:
        missing_classes = [
            class_name
            for class_name in CLASS_NAMES
            if len(structure[split_name].get(class_name, [])) == 0
        ]
        if missing_classes:
            raise FileNotFoundError(
                f"Split '{split_name}' is missing class folder(s): {', '.join(missing_classes)}"
            )

    return {
        split_name: {class_name: sorted(paths) for class_name, paths in class_dirs.items()}
        for split_name, class_dirs in structure.items()
    }


def list_image_files(directory: Path) -> list[Path]:
    """List supported image files under a directory recursively."""

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_split_samples(
    split_name: str,
    class_directories: dict[str, list[Path]],
) -> list[ImageSample]:
    """Build ImageSample records for one split."""

    samples: list[ImageSample] = []

    for class_name in CLASS_NAMES:
        directories = class_directories.get(class_name, [])
        label = CLASS_TO_IDX[class_name]

        for directory in directories:
            image_paths = list_image_files(directory)
            for image_path in image_paths:
                samples.append(
                    ImageSample(
                        path=image_path,
                        label=label,
                        class_name=class_name,
                        split_name=split_name,
                        source_directory=directory,
                    )
                )

    if not samples:
        raise RuntimeError(f"No images were discovered for split '{split_name}'.")

    return sorted(samples, key=lambda sample: str(sample.path).lower())


def build_transforms(
    image_size: tuple[int, int] = TRAINING_CONFIG.image_size,
) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """Create the training, validation, and test transforms.

    For the full profile we use a strong augmentation pipeline that pushes
    validation accuracy toward 95-99%:
      - TrivialAugmentWide (AutoAugment-style, very effective)
      - RandomErasing to regularise against background noise
      - Center-crop eval to stay consistent with EfficientNet-B3 paper

    For the fast/CPU profile we keep a lighter pipeline so training still
    completes in a reasonable time.
    """

    is_fast = TRAINING_CONFIG.profile == "fast"
    # Use slightly larger intermediate size then center-crop, standard for EfficientNet.
    h, w = image_size
    resize_size = (int(h * 256 / 224), int(w * 256 / 224))

    if is_fast:
        train_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=AUGMENTATION_CONFIG.horizontal_flip_probability),
                transforms.RandomRotation(
                    degrees=AUGMENTATION_CONFIG.rotation_degrees,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(
                    brightness=AUGMENTATION_CONFIG.brightness,
                    contrast=AUGMENTATION_CONFIG.contrast,
                    saturation=AUGMENTATION_CONFIG.saturation,
                    hue=AUGMENTATION_CONFIG.hue,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            ]
        )
    else:
        # Full profile: strong augmentation for 95%+ accuracy target
        train_transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomCrop(image_size, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=AUGMENTATION_CONFIG.horizontal_flip_probability),
                transforms.RandomVerticalFlip(p=AUGMENTATION_CONFIG.vertical_flip_probability),
                transforms.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
                transforms.RandomRotation(
                    degrees=AUGMENTATION_CONFIG.rotation_degrees,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(
                    brightness=AUGMENTATION_CONFIG.brightness,
                    contrast=AUGMENTATION_CONFIG.contrast,
                    saturation=AUGMENTATION_CONFIG.saturation,
                    hue=AUGMENTATION_CONFIG.hue,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=AUGMENTATION_CONFIG.affine_translate,
                    scale=AUGMENTATION_CONFIG.affine_scale,
                    shear=AUGMENTATION_CONFIG.affine_shear,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=AUGMENTATION_CONFIG.gaussian_blur_kernel_size,
                            sigma=AUGMENTATION_CONFIG.gaussian_blur_sigma,
                        )
                    ],
                    p=AUGMENTATION_CONFIG.gaussian_blur_probability,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                # RandomErasing regularises the model against background-heavy images
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform, eval_transform


def compute_split_statistics(samples: Iterable[ImageSample], split_name: str) -> SplitStatistics:
    """Compute class counts and imbalance ratio for one split."""

    class_counts = {class_name: 0 for class_name in CLASS_NAMES}
    for sample in samples:
        class_counts[sample.class_name] += 1

    total_images = sum(class_counts.values())
    non_zero_counts = [count for count in class_counts.values() if count > 0]
    imbalance_ratio = (
        max(non_zero_counts) / min(non_zero_counts) if len(non_zero_counts) == 2 else math.inf
    )

    return SplitStatistics(
        split_name=split_name,
        class_counts=class_counts,
        total_images=total_images,
        imbalance_ratio=imbalance_ratio,
    )


def print_dataset_statistics(statistics: dict[str, SplitStatistics]) -> None:
    """Print an easy-to-read summary for all dataset splits."""

    print("\nDataset statistics")
    print("-" * 72)
    for split_name in SPLIT_NAMES:
        stats = statistics[split_name]
        ratio_text = (
            f"{stats.imbalance_ratio:.2f}:1"
            if math.isfinite(stats.imbalance_ratio)
            else "undefined"
        )
        print(
            f"{split_name:<8} "
            f"anemic={stats.class_counts['anemic']:<5} "
            f"non-anemic={stats.class_counts['non-anemic']:<5} "
            f"total={stats.total_images:<5} "
            f"imbalance={ratio_text}"
        )
    print("-" * 72)


def create_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Create a weighted sampler so minority-class images are sampled more often."""

    label_counts = Counter(labels)
    sample_count = len(labels)
    class_weights = {
        label: sample_count / (len(label_counts) * count)
        for label, count in label_counts.items()
    }
    sample_weights = torch.DoubleTensor([class_weights[label] for label in labels])

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def seed_worker(worker_id: int) -> None:
    """Keep data loading reproducible across workers."""

    worker_seed = TRAINING_CONFIG.random_seed + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_dataloaders(
    dataset_root: Path = DATASET_ROOT,
    batch_size: int = TRAINING_CONFIG.batch_size,
    num_workers: int = TRAINING_CONFIG.num_workers,
    show_statistics: bool = True,
) -> DataLoaderBundle:
    """Build train/valid/test DataLoaders with automatic dataset discovery."""

    structure = discover_dataset_structure(dataset_root=dataset_root)
    train_transform, valid_transform, test_transform = build_transforms()

    train_samples = collect_split_samples("train", structure["train"])
    valid_samples = collect_split_samples("valid", structure["valid"])
    test_samples = collect_split_samples("test", structure["test"])

    statistics = {
        "train": compute_split_statistics(train_samples, "train"),
        "valid": compute_split_statistics(valid_samples, "valid"),
        "test": compute_split_statistics(test_samples, "test"),
    }

    if show_statistics:
        print_dataset_statistics(statistics)

    train_dataset = AnemiaImageDataset(train_samples, transform=train_transform)
    valid_dataset = AnemiaImageDataset(valid_samples, transform=valid_transform)
    test_dataset = AnemiaImageDataset(test_samples, transform=test_transform)

    train_sampler = create_weighted_sampler(train_dataset.targets)
    generator = torch.Generator()
    generator.manual_seed(TRAINING_CONFIG.random_seed)

    loader_extras: dict[str, object] = {}
    if num_workers > 0:
        loader_extras["persistent_workers"] = True
        loader_extras["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=TRAINING_CONFIG.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        **loader_extras,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=TRAINING_CONFIG.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        **loader_extras,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=TRAINING_CONFIG.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        **loader_extras,
    )

    return DataLoaderBundle(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        statistics=statistics,
        class_to_idx=CLASS_TO_IDX,
        idx_to_class=IDX_TO_CLASS,
    )


if __name__ == "__main__":
    create_dataloaders()
