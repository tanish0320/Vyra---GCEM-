"""Model utilities for AnemiaVision AI."""

from __future__ import annotations

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


def load_checkpoint(
    checkpoint_path: Path = BEST_CHECKPOINT_PATH,
    device: torch.device | str = "cpu",
    pretrained_fallback: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved checkpoint and restore the EfficientNet-B3 weights."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    architecture = checkpoint.get("architecture", "efficientnet_b3")
    model = build_model(pretrained=pretrained_fallback, architecture=architecture)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint


def latest_available_checkpoint() -> Path | None:
    """Return the most useful checkpoint that currently exists on disk."""

    for checkpoint_path in (BEST_CHECKPOINT_PATH, LATEST_CHECKPOINT_PATH):
        if checkpoint_path.exists():
            return checkpoint_path
    return None
