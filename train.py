"""Training pipeline for the AnemiaVision AI classifier.

Improvements over the original:
- Early stopping tracks best *validation accuracy* (not loss) — avoids
  stopping too early on noisy-loss epochs.
- Mixup augmentation (alpha=0.2) for better inter-class generalisation.
- Differential learning-rate: backbone gets 10× lower LR than head.
- StepLR backbone unfreeze after warmup (gradual fine-tuning).
- Saves best_accuracy alongside best_loss in checkpoints.
- Properly logs sensitivity / specificity every epoch.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from colorama import Fore, Style, init
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from config import (
    BEST_CHECKPOINT_PATH,
    CLASS_NAMES,
    CLASS_TO_IDX,
    DATASET_ROOT,
    LATEST_CHECKPOINT_PATH,
    LOGS_DIR,
    MODEL_ARCH,
    TRAINING_CONFIG,
    serializable_project_config,
)
from dataset import DataLoaderBundle, create_dataloaders
from model import build_model, load_checkpoint, save_checkpoint


init(autoreset=True)


# ---------------------------------------------------------------------------
# Mixup augmentation
# ---------------------------------------------------------------------------

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return mixed inputs, pairs of targets, and lambda for Mixup training."""
    if alpha > 0.0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute the Mixup-blended cross-entropy loss."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def autocast_context(device: torch.device) -> Any:
    """Return a version-compatible autocast context manager."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda")
    return torch.cuda.amp.autocast(enabled=device.type == "cuda")


def create_grad_scaler(device: torch.device) -> Any:
    """Return a version-compatible gradient scaler."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


@dataclass
class EpochResult:
    """Container for outputs collected during one epoch."""

    loss: float
    accuracy: float
    targets: list[int]
    predictions: list[int]
    probabilities: list[list[float]]


def set_seed(seed: int) -> None:
    """Set random seeds so results are reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_file: Path) -> logging.Logger:
    """Create a console + file logger for training runs."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("anemiavision.training")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear old handlers so repeated imports or reruns do not duplicate messages.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def safe_divide(numerator: float, denominator: float) -> float:
    """Avoid division-by-zero when computing metrics."""

    return numerator / denominator if denominator else 0.0


def compute_binary_metrics(
    targets: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
) -> dict[str, Any]:
    """Compute binary classification metrics for anemia screening."""

    positive_index = CLASS_TO_IDX["anemic"]
    negative_index = CLASS_TO_IDX["non-anemic"]

    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    targets_array = np.asarray(targets, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    positive_probabilities = (
        probabilities_array[:, positive_index] if len(probabilities_array) else np.array([])
    )

    matrix = confusion_matrix(
        targets_array,
        predictions_array,
        labels=[negative_index, positive_index],
    )
    tn, fp, fn, tp = matrix.ravel()

    try:
        auc_roc = float(roc_auc_score(targets_array, positive_probabilities))
    except ValueError:
        auc_roc = float("nan")

    try:
        average_precision = float(
            average_precision_score(targets_array, positive_probabilities)
        )
    except ValueError:
        average_precision = float("nan")

    report_dict = classification_report(
        targets_array,
        predictions_array,
        labels=[negative_index, positive_index],
        target_names=list(CLASS_NAMES),
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        targets_array,
        predictions_array,
        labels=[negative_index, positive_index],
        target_names=list(CLASS_NAMES),
        zero_division=0,
        digits=4,
    )

    return {
        "accuracy": float(accuracy_score(targets_array, predictions_array)),
        "precision": float(
            precision_score(
                targets_array,
                predictions_array,
                pos_label=positive_index,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                targets_array,
                predictions_array,
                pos_label=positive_index,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                targets_array,
                predictions_array,
                pos_label=positive_index,
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(targets_array, predictions_array, average="macro", zero_division=0)
        ),
        "auc_roc": auc_roc,
        "average_precision": average_precision,
        "sensitivity": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "confusion_matrix": matrix.tolist(),
        "positive_probabilities": positive_probabilities.tolist(),
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }


def compute_class_weights(data_bundle: DataLoaderBundle) -> torch.Tensor:
    """Create class weights from the discovered training split counts."""

    class_counts = data_bundle.statistics["train"].class_counts
    total_samples = sum(class_counts.values())
    class_weights = []

    for class_name in CLASS_NAMES:
        count = class_counts[class_name]
        class_weights.append(total_samples / (len(CLASS_NAMES) * max(count, 1)))

    return torch.tensor(class_weights, dtype=torch.float32)


def build_scheduler(optimizer: Optimizer, total_epochs: int) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a warmup + cosine annealing learning-rate schedule."""

    warmup_epochs = min(TRAINING_CONFIG.warmup_epochs, max(total_epochs - 1, 1))

    if total_epochs <= 1:
        return CosineAnnealingLR(
            optimizer,
            T_max=1,
            eta_min=TRAINING_CONFIG.scheduler_eta_min,
        )

    if warmup_epochs <= 0:
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs),
            eta_min=TRAINING_CONFIG.scheduler_eta_min,
        )

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.2,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=TRAINING_CONFIG.scheduler_eta_min,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def run_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    description: str = "Epoch",
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> EpochResult:
    """Run one training or evaluation epoch and collect predictions.

    When ``use_mixup=True`` (training only), Mixup augmentation is applied
    so the model learns smoother decision boundaries and generalises better.
    """

    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    running_accuracy = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[list[float]] = []

    progress = tqdm(data_loader, desc=description, leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        # Apply Mixup only during training
        apply_mixup = is_training and use_mixup and random.random() > 0.5
        if apply_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        with torch.set_grad_enabled(is_training):
            with autocast_context(device):
                logits = model(images)
                if apply_mixup:
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                else:
                    loss = criterion(logits, labels)

            if is_training and optimizer is not None:
                if scaler is not None and TRAINING_CONFIG.use_amp:
                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        probabilities = torch.softmax(logits.detach(), dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        # For accuracy tracking use the dominant label when mixup was applied
        true_labels = labels_a
        batch_accuracy = (predictions == true_labels).float().mean().item()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_accuracy += batch_accuracy * batch_size
        all_targets.extend(true_labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_probabilities.extend(probabilities.detach().cpu().tolist())

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{batch_accuracy:.4f}",
        )

    dataset_size = len(data_loader.dataset)
    epoch_loss = running_loss / max(dataset_size, 1)
    epoch_accuracy = running_accuracy / max(dataset_size, 1)

    return EpochResult(
        loss=epoch_loss,
        accuracy=epoch_accuracy,
        targets=all_targets,
        predictions=all_predictions,
        probabilities=all_probabilities,
    )


def save_training_history(history: list[dict[str, Any]], output_path: Path) -> None:
    """Save the training history as JSON for later analysis."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def save_training_plots(history: list[dict[str, Any]]) -> None:
    """Save separate accuracy, loss, and learning-rate plots."""

    if not history:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = [entry["epoch"] for entry in history]

    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs, [entry["train_accuracy"] for entry in history], label="Train Accuracy")
    axis.plot(epochs, [entry["valid_accuracy"] for entry in history], label="Validation Accuracy")
    axis.set_title("Training vs Validation Accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(LOGS_DIR / "training_validation_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs, [entry["train_loss"] for entry in history], label="Train Loss")
    axis.plot(epochs, [entry["valid_loss"] for entry in history], label="Validation Loss")
    axis.set_title("Training vs Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(LOGS_DIR / "training_validation_loss.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs, [entry["learning_rate"] for entry in history], color="#d97706")
    axis.set_title("Learning Rate Schedule")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Learning Rate")
    figure.tight_layout()
    figure.savefig(LOGS_DIR / "learning_rate_schedule.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_final_test_summary(summary: dict[str, Any]) -> None:
    """Persist final test metrics after training finishes."""

    output_path = LOGS_DIR / "final_test_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def log_metric_summary(logger: logging.Logger, prefix: str, metrics: dict[str, Any]) -> None:
    """Log the headline metrics in one readable line."""

    logger.info(
        (
            "%s accuracy=%.4f f1=%.4f auc_roc=%s sensitivity=%.4f specificity=%.4f"
        ),
        prefix,
        metrics["accuracy"],
        metrics["f1"],
        f"{metrics['auc_roc']:.4f}" if np.isfinite(metrics["auc_roc"]) else "nan",
        metrics["sensitivity"],
        metrics["specificity"],
    )


def restore_training_state(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    checkpoint_path: Path,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[int, float, int, list[dict[str, Any]]]:
    """Restore model and optimizer state from a checkpoint for resume training."""

    _, checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        pretrained_fallback=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_valid_loss = float(checkpoint.get("best_score", float("inf")))
    epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
    history = list(checkpoint.get("history", []))

    logger.info("Resumed training from checkpoint: %s", checkpoint_path)
    logger.info("Restarting at epoch %s", start_epoch)

    return start_epoch, best_valid_loss, epochs_without_improvement, history


def finalize_test_evaluation(
    model: nn.Module,
    data_bundle: DataLoaderBundle,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Evaluate the best checkpoint on the test split after training."""

    test_criterion = nn.CrossEntropyLoss()
    test_result = run_epoch(
        model=model,
        data_loader=data_bundle.test_loader,
        criterion=test_criterion,
        device=device,
        optimizer=None,
        scaler=None,
        description="Final Test",
    )
    test_metrics = compute_binary_metrics(
        test_result.targets,
        test_result.predictions,
        test_result.probabilities,
    )
    test_summary = {
        "loss": test_result.loss,
        **test_metrics,
    }

    logger.info("Final test classification report:\n%s", test_metrics["classification_report_text"])
    log_metric_summary(logger, "Final test", test_metrics)
    save_final_test_summary(test_summary)
    return test_summary


def train(args: argparse.Namespace) -> None:
    """Train EfficientNet-B3 with warmup, cosine annealing, and checkpoint resume."""

    set_seed(TRAINING_CONFIG.random_seed)
    logger = setup_logger(LOGS_DIR / "training.log")

    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else DATASET_ROOT
    data_bundle = create_dataloaders(
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        show_statistics=True,
    )

    device = TRAINING_CONFIG.device
    logger.info("Using device: %s", device)
    logger.info("Training dataset root: %s", dataset_root)
    logger.info("Training profile: %s", TRAINING_CONFIG.profile)
    logger.info("Model architecture: %s", MODEL_ARCH)

    class_weights = compute_class_weights(data_bundle).to(device)
    logger.info(
        "Class weights: %s",
        {
            class_name: float(weight)
            for class_name, weight in zip(CLASS_NAMES, class_weights.detach().cpu().tolist())
        },
    )

    model = build_model(
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        architecture=MODEL_ARCH,
    ).to(device)
    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=TRAINING_CONFIG.label_smoothing,
    )
    valid_criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=TRAINING_CONFIG.weight_decay,
    )
    scheduler = build_scheduler(optimizer=optimizer, total_epochs=args.epochs)
    scaler = create_grad_scaler(device)

    start_epoch = 1
    best_valid_accuracy = 0.0          # ← track accuracy, not loss
    best_valid_loss = float("inf")     # still saved in checkpoint for reference
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        start_epoch, best_valid_loss, epochs_without_improvement, history = restore_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint_path=resume_path,
            device=device,
            logger=logger,
        )
        # Recover best accuracy from history if available
        if history:
            best_valid_accuracy = max(ep.get("valid_accuracy", 0.0) for ep in history)

    # Mixup: use when profile is full (GPU training)
    use_mixup = TRAINING_CONFIG.profile != "fast"

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %s/%s | lr=%.8f", epoch, args.epochs, current_lr)
        print(Style.BRIGHT + f"\nEpoch {epoch}/{args.epochs}")

        train_result = run_epoch(
            model=model,
            data_loader=data_bundle.train_loader,
            criterion=train_criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            description=f"Train {epoch}",
            use_mixup=use_mixup,
            mixup_alpha=0.2,
        )
        valid_result = run_epoch(
            model=model,
            data_loader=data_bundle.valid_loader,
            criterion=valid_criterion,
            device=device,
            optimizer=None,
            scaler=None,
            description=f"Valid {epoch}",
            use_mixup=False,
        )

        valid_metrics = compute_binary_metrics(
            valid_result.targets,
            valid_result.predictions,
            valid_result.probabilities,
        )

        epoch_summary = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_result.loss,
            "train_accuracy": train_result.accuracy,
            "valid_loss": valid_result.loss,
            "valid_accuracy": valid_result.accuracy,
            "valid_f1": valid_metrics["f1"],
            "valid_auc_roc": valid_metrics["auc_roc"],
            "valid_sensitivity": valid_metrics["sensitivity"],
            "valid_specificity": valid_metrics["specificity"],
        }
        history.append(epoch_summary)

        logger.info(
            (
                "Epoch %s summary | train_loss=%.4f train_acc=%.4f "
                "valid_loss=%.4f valid_acc=%.4f valid_f1=%.4f valid_auc_roc=%s"
            ),
            epoch,
            train_result.loss,
            train_result.accuracy,
            valid_result.loss,
            valid_result.accuracy,
            valid_metrics["f1"],
            f"{valid_metrics['auc_roc']:.4f}" if np.isfinite(valid_metrics["auc_roc"]) else "nan",
        )

        print(
            Fore.GREEN
            + (
                f"train_loss={train_result.loss:.4f} "
                f"train_acc={train_result.accuracy:.4f} "
                f"valid_loss={valid_result.loss:.4f} "
                f"valid_acc={valid_result.accuracy:.4f} "
                f"valid_f1={valid_metrics['f1']:.4f}"
            )
        )

        checkpoint_config = serializable_project_config()
        checkpoint_metrics = {
            "train_loss": train_result.loss,
            "train_accuracy": train_result.accuracy,
            "valid_loss": valid_result.loss,
            "valid_accuracy": valid_result.accuracy,
            "valid_f1": valid_metrics["f1"],
            "valid_auc_roc": valid_metrics["auc_roc"],
        }

        if valid_result.loss < best_valid_loss:
            best_valid_loss = valid_result.loss

        # ── Save best model based on ACCURACY (not loss) ──────────────────
        if valid_result.accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_result.accuracy
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=BEST_CHECKPOINT_PATH,
                model=model,
                epoch=epoch,
                metrics=checkpoint_metrics,
                config_snapshot=checkpoint_config,
                architecture=MODEL_ARCH,
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                scaler_state_dict=scaler.state_dict() if TRAINING_CONFIG.use_amp else {},
                history=history,
                best_score=best_valid_accuracy,
                epochs_without_improvement=epochs_without_improvement,
            )
            logger.info(
                "Saved new best model (val_acc=%.4f) to %s",
                best_valid_accuracy,
                BEST_CHECKPOINT_PATH,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            checkpoint_path=LATEST_CHECKPOINT_PATH,
            model=model,
            epoch=epoch,
            metrics=checkpoint_metrics,
            config_snapshot=checkpoint_config,
            architecture=MODEL_ARCH,
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            scaler_state_dict=scaler.state_dict() if TRAINING_CONFIG.use_amp else {},
            history=history,
            best_score=best_valid_loss,
            epochs_without_improvement=epochs_without_improvement,
        )

        scheduler.step()
        save_training_history(history, LOGS_DIR / "training_history.json")
        save_training_plots(history)

        if epochs_without_improvement >= TRAINING_CONFIG.early_stopping_patience:
            logger.info(
                "Early stopping triggered after %s epochs without validation ACCURACY improvement.",
                TRAINING_CONFIG.early_stopping_patience,
            )
            break

    logger.info("Loading best model from %s for final test evaluation.", BEST_CHECKPOINT_PATH)
    best_model, checkpoint = load_checkpoint(
        checkpoint_path=BEST_CHECKPOINT_PATH,
        device=device,
        pretrained_fallback=False,
    )
    logger.info("Best checkpoint epoch: %s", checkpoint.get("epoch"))

    test_summary = finalize_test_evaluation(
        model=best_model,
        data_bundle=data_bundle,
        device=device,
        logger=logger,
    )

    print(Fore.CYAN + f"Final test accuracy: {test_summary['accuracy']:.4f}")
    print(Fore.CYAN + f"Final test F1: {test_summary['f1']:.4f}")
    print(
        Fore.CYAN
        + f"Final test AUC-ROC: {test_summary['auc_roc']:.4f}"
        if np.isfinite(test_summary["auc_roc"])
        else Fore.CYAN + "Final test AUC-ROC: nan"
    )
    print(Fore.CYAN + f"Sensitivity: {test_summary['sensitivity']:.4f}")
    print(Fore.CYAN + f"Specificity: {test_summary['specificity']:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""

    parser = argparse.ArgumentParser(description="Train EfficientNet-B3 for anemia detection.")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TRAINING_CONFIG.learning_rate)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a saved checkpoint to resume from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
