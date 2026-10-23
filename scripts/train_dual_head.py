"""Training script for Dual-Head EfficientNet-B3 on VYRA Leakage-Safe Splits.

Features:
- Joint learning objective: L_total = 0.7 * L_CE + 0.3 * L_MSE
- Masked loss for continuous Hb regression: samples without lab Hb are masked from L_MSE
- Mixed precision (AMP) with torch.cuda.amp.GradScaler
- Checkpoint selection and early stopping based STRICTLY on validation split metrics
- Untouched test split is reserved exclusively for final evaluation
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from src.data.manifest_dataset import ManifestDataset, create_manifest_dataloaders
from src.models.dual_head_model import DualHeadEfficientNetB3, MaskedDualLoss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"


def compute_class_weights(train_csv: Path) -> torch.Tensor:
    """Compute inverse frequency class weights to balance Cross-Entropy."""
    import pandas as pd
    df = pd.read_csv(train_csv)
    labels = df["label"].values
    counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(counts) * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MaskedDualLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_mse = 0.0
    correct = 0
    total_samples = 0
    hb_samples = 0
    hb_abs_errors = []

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        body_part = batch["body_part"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        hb = batch["hb"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits, hb_pred = model(images, features, body_part)
            loss, ce, mse = criterion(logits, hb_pred, labels, hb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_ce += ce.item() * batch_size
        total_mse += mse.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += batch_size

        # Track Hb regression metrics on samples with valid ground truth
        valid_mask = ~torch.isnan(hb) & (hb > 0.0)
        if valid_mask.sum() > 0:
            diffs = torch.abs(hb_pred[valid_mask] - hb[valid_mask]).detach().cpu().numpy()
            hb_abs_errors.extend(diffs.flatten())
            hb_samples += int(valid_mask.sum().item())

    mae = float(np.mean(hb_abs_errors)) if hb_abs_errors else 0.0
    return {
        "loss": total_loss / total_samples,
        "ce_loss": total_ce / total_samples,
        "mse_loss": total_mse / total_samples,
        "accuracy": correct / total_samples,
        "hb_mae": mae,
        "hb_samples": hb_samples,
    }


def evaluate_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MaskedDualLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_mse = 0.0
    correct = 0
    total_samples = 0
    hb_abs_errors = []
    hb_sq_errors = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            features = batch["features"].to(device, non_blocking=True)
            body_part = batch["body_part"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            hb = batch["hb"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                logits, hb_pred = model(images, features, body_part)
                loss, ce, mse = criterion(logits, hb_pred, labels, hb)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_ce += ce.item() * batch_size
            total_mse += mse.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += batch_size

            valid_mask = ~torch.isnan(hb) & (hb > 0.0)
            if valid_mask.sum() > 0:
                diffs = torch.abs(hb_pred[valid_mask] - hb[valid_mask]).detach().cpu().numpy().flatten()
                hb_abs_errors.extend(diffs)
                hb_sq_errors.extend(diffs ** 2)

    mae = float(np.mean(hb_abs_errors)) if hb_abs_errors else 0.0
    rmse = float(np.sqrt(np.mean(hb_sq_errors))) if hb_sq_errors else 0.0

    return {
        "loss": total_loss / total_samples,
        "ce_loss": total_ce / total_samples,
        "mse_loss": total_mse / total_samples,
        "accuracy": correct / total_samples,
        "hb_mae": mae,
        "hb_rmse": rmse,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DualHead EfficientNet-B3 model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "splits" / "validation.csv"

    class_weights = compute_class_weights(train_csv).to(device)
    print(f"Inverse class weights: {class_weights.tolist()}")

    loaders = create_manifest_dataloaders(
        train_manifest=train_csv,
        val_manifest=val_csv,
        batch_size=args.batch_size,
        num_workers=0,  # 0 for Windows multiprocessing safety
    )

    model = DualHeadEfficientNetB3(pretrained=True, freeze_backbone=False).to(device)
    criterion = MaskedDualLoss(alpha=0.70, beta=0.30, class_weights=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_checkpoint_path = CHECKPOINTS_DIR / "best_dual_head.pth"
    history = []
    epochs_no_improve = 0

    print("\n" + "=" * 65)
    print("STARTING DUAL-HEAD TRAINING ON RTX 4060 GPU")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, loaders["train"], criterion, optimizer, scaler, device)
        val_metrics = evaluate_split(model, loaders["val"], criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} (Acc: {train_metrics['accuracy']*100:.1f}%, Hb MAE: {train_metrics['hb_mae']:.2f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} (Acc: {val_metrics['accuracy']*100:.1f}%, Hb MAE: {val_metrics['hb_mae']:.2f})"
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": float(scheduler.get_last_lr()[0]),
            "time_s": elapsed,
        }
        history.append(record)

        # Validation-only model selection
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": val_metrics,
                "has_validated_regression": True,
                "temperature": 1.0,
                "architecture": "dual_head_efficientnet_b3",
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint: {best_checkpoint_path} (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience: {args.patience}).")
                break

    # Save training history
    with open(LOGS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nWrote training history to {LOGS_DIR / 'training_history.json'}")
    print(f"Best model checkpoint saved at: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
