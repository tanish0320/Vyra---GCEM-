"""Model utilities for AnemiaVision AI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    efficientnet_b0,
    efficientnet_b3,
)

from config import BEST_CHECKPOINT_PATH, CLASS_NAMES, LATEST_CHECKPOINT_PATH, MODEL_ARCH

LFS_POINTER_HEADER = "version https://git-lfs.github.com/spec/v1"
MIN_VALID_CHECKPOINT_BYTES = 1024


_MODEL_REGISTRY = {
    "efficientnet_b0": {
        "builder": efficientnet_b0,
        "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "classifier_features": 1280,
    },
    "efficientnet_b3": {
        "builder": efficientnet_b3,
        "weights": EfficientNet_B3_Weights.IMAGENET1K_V1,
        "classifier_features": 1536,
    },
}


def resolve_architecture(architecture: str | None) -> str:
    """Normalize the architecture name and fall back to defaults."""

    if architecture:
        normalized = architecture.strip().lower()
    else:
        normalized = MODEL_ARCH

    if normalized not in _MODEL_REGISTRY:
        options = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unsupported architecture '{architecture}'. Options: {options}")

    return normalized


def build_model(
    num_classes: int = len(CLASS_NAMES),
    pretrained: bool = True,
    freeze_backbone: bool = False,
    architecture: str | None = None,
) -> nn.Module:
    """Build an EfficientNet classifier with the Stage 2 custom head."""

    resolved_architecture = resolve_architecture(architecture)
    spec = _MODEL_REGISTRY[resolved_architecture]
    weights = spec["weights"] if pretrained else None
    model = spec["builder"](weights=weights)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    classifier_in_features = spec["classifier_features"]
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
    # Stronger head: two hidden layers with BatchNorm + GELU + high dropout
    # to prevent over-fitting on small eye-image datasets.
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(classifier_in_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(p=0.45),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(p=0.35),
        nn.Linear(256, num_classes),
    )

    return model


def checkpoint_payload(
    model: nn.Module,
    epoch: int,
    metrics: dict[str, float],
    config_snapshot: dict[str, Any],
    architecture: str | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    scaler_state_dict: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    best_score: float | None = None,
    epochs_without_improvement: int | None = None,
) -> dict[str, Any]:
    """Package model weights with the metadata needed for reproducibility."""

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": config_snapshot,
        "class_names": list(CLASS_NAMES),
        "architecture": resolve_architecture(architecture),
        "history": history or [],
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = scheduler_state_dict
    if scaler_state_dict is not None:
        payload["scaler_state_dict"] = scaler_state_dict
    if best_score is not None:
        payload["best_score"] = best_score
    if epochs_without_improvement is not None:
        payload["epochs_without_improvement"] = epochs_without_improvement
    return payload


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    metrics: dict[str, float],
    config_snapshot: dict[str, Any],
    checkpoint_path: Path,
    architecture: str | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    scaler_state_dict: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    best_score: float | None = None,
    epochs_without_improvement: int | None = None,
) -> None:
    """Save a training checkpoint to disk."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model=model,
            epoch=epoch,
            metrics=metrics,
            config_snapshot=config_snapshot,
            architecture=architecture,
            optimizer_state_dict=optimizer_state_dict,
            scheduler_state_dict=scheduler_state_dict,
            scaler_state_dict=scaler_state_dict,
            history=history,
            best_score=best_score,
            epochs_without_improvement=epochs_without_improvement,
        ),
        checkpoint_path,
    )


def is_lfs_pointer_file(checkpoint_path: str | Path | None) -> bool:
    """Return True if the file at checkpoint_path is a Git LFS pointer text file."""

    if checkpoint_path is None:
        return False
    path = Path(checkpoint_path)
    if not path.exists() or not path.is_file():
        return False

    try:
        with path.open("rb") as handle:
            prefix = handle.read(256)
    except OSError:
        return False

    text_prefix = prefix.decode("utf-8", errors="ignore").lstrip()
    if text_prefix.startswith(LFS_POINTER_HEADER):
        return True
    if "git-lfs.github.com/spec" in text_prefix and "oid sha256:" in text_prefix:
        return True
    return False


def inspect_checkpoint_file(checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    """Inspect a checkpoint path and report existence, LFS status, and validity."""

    target = Path(checkpoint_path) if checkpoint_path is not None else latest_available_checkpoint(require_valid=False)
    if target is None or not target.exists():
        return {
            "path": str(target) if target is not None else None,
            "checkpoint_exists": False,
            "checkpoint_is_lfs_pointer": False,
            "checkpoint_valid": False,
            "size_bytes": 0,
            "failure_category": "missing_checkpoint",
            "reason": "No trained checkpoint file was found on disk.",
        }

    try:
        size_bytes = target.stat().st_size
    except OSError as exc:
        return {
            "path": str(target),
            "checkpoint_exists": True,
            "checkpoint_is_lfs_pointer": False,
            "checkpoint_valid": False,
            "size_bytes": 0,
            "failure_category": "unreadable_checkpoint",
            "reason": f"Checkpoint file could not be stat'd: {exc}",
        }

    if is_lfs_pointer_file(target):
        return {
            "path": str(target),
            "checkpoint_exists": True,
            "checkpoint_is_lfs_pointer": True,
            "checkpoint_valid": False,
            "size_bytes": size_bytes,
            "failure_category": "lfs_pointer_checkpoint",
            "reason": (
                f"Checkpoint file '{target.name}' ({size_bytes} bytes) is a Git LFS pointer text file, "
                "not actual PyTorch model weights."
            ),
        }

    if size_bytes < MIN_VALID_CHECKPOINT_BYTES:
        return {
            "path": str(target),
            "checkpoint_exists": True,
            "checkpoint_is_lfs_pointer": False,
            "checkpoint_valid": False,
            "size_bytes": size_bytes,
            "failure_category": "corrupt_checkpoint",
            "reason": f"Checkpoint file '{target.name}' ({size_bytes} bytes) is too small to be a valid PyTorch checkpoint.",
        }

    return {
        "path": str(target),
        "checkpoint_exists": True,
        "checkpoint_is_lfs_pointer": False,
        "checkpoint_valid": True,
        "size_bytes": size_bytes,
        "failure_category": None,
        "reason": None,
    }


def load_checkpoint(
    checkpoint_path: Path = BEST_CHECKPOINT_PATH,
    device: torch.device | str = "cpu",
    pretrained_fallback: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved checkpoint and restore the EfficientNet-B3 weights."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    inspection = inspect_checkpoint_file(checkpoint_path)
    if inspection["checkpoint_is_lfs_pointer"]:
        raise RuntimeError(inspection["reason"])
    if not inspection["checkpoint_valid"]:
        raise RuntimeError(inspection["reason"] or f"Invalid checkpoint file: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    is_dual_head = False
    if isinstance(state_dict, dict):
        is_dual_head = (
            "classifier_head.0.weight" in state_dict
            or "classifier_head.4.weight" in state_dict
            or "regression_head.0.weight" in state_dict
            or (isinstance(checkpoint, dict) and checkpoint.get("has_validated_regression") is not None)
        )

    if is_dual_head:
        try:
            repo_root = Path(__file__).resolve().parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from src.models.dual_head_model import DualHeadEfficientNetB3
            model = DualHeadEfficientNetB3(pretrained=False)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model, checkpoint
        except ImportError:
            pass

    architecture = checkpoint.get("architecture", "efficientnet_b3") if isinstance(checkpoint, dict) else "efficientnet_b3"
    model = build_model(pretrained=pretrained_fallback, architecture=architecture)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint


def latest_available_checkpoint(require_valid: bool = False) -> Path | None:
    """Return the most useful checkpoint that currently exists on disk."""

    for checkpoint_path in (BEST_CHECKPOINT_PATH, LATEST_CHECKPOINT_PATH):
        if checkpoint_path.exists():
            if require_valid and not inspect_checkpoint_file(checkpoint_path)["checkpoint_valid"]:
                continue
            return checkpoint_path
    return None
