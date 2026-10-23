"""PyTorch Dataset loader backed by reproducible split manifests.

Features:
- Reads train.csv, validation.csv, or test.csv
- Deterministic optical preprocessing (Grey-World, Specular Glare inpainting, HSV-CLAHE, ITA)
- Extracted 27-dim colorimetric feature vector
- Body part one-hot encoding
- Handles continuous Hb target or sets NaN when missing
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from src.data.normalization_adapter import preprocess_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ManifestDataset(Dataset):
    """Dataset driven by leakage-safe split manifests."""

    def __init__(
        self,
        manifest_csv: Path | str,
        image_size: tuple[int, int] = (224, 224),
        is_training: bool = False,
        apply_normalization: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_csv)
        self.image_size = image_size
        self.is_training = is_training
        self.apply_normalization = apply_normalization
        self.samples = self._load_manifest(self.manifest_path)

        # Base tensor conversion & normalization
        self.to_tensor_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_manifest(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        records = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        return records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.samples[idx]
        file_path = PROJECT_ROOT / row["file_path"]
        body_part = row.get("modality", "fingernail")
        class_label = int(row.get("label", 0))

        # Handle lab Hb label
        raw_hb = row.get("lab_hb_g_dl", "").strip()
        if raw_hb:
            try:
                hb_val = float(raw_hb)
            except ValueError:
                hb_val = float("nan")
        else:
            hb_val = float("nan")

        # Run optical preprocessing & feature extraction
        if self.apply_normalization:
            prep = preprocess_image(file_path, body_part=body_part)
            img_rgb = prep.normalized_image_rgb
            features = prep.feature_vector
            body_enc = prep.body_part_encoding
            ita = prep.ita_angle
            warnings = prep.quality_warnings
        else:
            with Image.open(file_path) as img:
                img_rgb = np.array(img.convert("RGB"))
            features = np.zeros(27, dtype=np.float32)
            body_enc = np.array([0.0, 1.0] if "nail" in body_part else [1.0, 0.0], dtype=np.float32)
            ita = 0.0
            warnings = []

        image_tensor = self.to_tensor_transform(img_rgb)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        body_tensor = torch.tensor(body_enc, dtype=torch.float32)
        hb_tensor = torch.tensor([hb_val], dtype=torch.float32)
        label_tensor = torch.tensor(class_label, dtype=torch.long)

        return {
            "image": image_tensor,
            "features": features_tensor,
            "body_part": body_tensor,
            "label": label_tensor,
            "hb": hb_tensor,
            "patient_id": row.get("patient_id", "unknown"),
            "dataset": row.get("dataset", "unknown"),
            "modality": body_part,
            "file_path": str(file_path),
            "ita": ita,
            "warnings": "; ".join(warnings) if warnings else "",
        }


def collate_manifest_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for ManifestDataset batches."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "features": torch.stack([b["features"] for b in batch]),
        "body_part": torch.stack([b["body_part"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "hb": torch.stack([b["hb"] for b in batch]),
        "patient_id": [b["patient_id"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
        "modality": [b["modality"] for b in batch],
        "file_path": [b["file_path"] for b in batch],
        "ita": [b["ita"] for b in batch],
        "warnings": [b["warnings"] for b in batch],
    }


def create_manifest_dataloaders(
    train_manifest: Path | str,
    val_manifest: Path | str,
    test_manifest: Optional[Path | str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create data loaders from manifests."""
    train_ds = ManifestDataset(train_manifest, is_training=True)
    val_ds = ManifestDataset(val_manifest, is_training=False)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_manifest_batch,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_manifest_batch,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    if test_manifest:
        test_ds = ManifestDataset(test_manifest, is_training=False)
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_manifest_batch,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders
