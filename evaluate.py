"""Evaluation pipeline for Stage 2 AnemiaVision AI checkpoints."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve

from config import (
    BEST_CHECKPOINT_PATH,
    CLASS_NAMES,
    CLASS_TO_IDX,
    DATASET_ROOT,
    EVALUATION_LOGS_DIR,
    IDX_TO_CLASS,
    TRAINING_CONFIG,
)
from dataset import AnemiaImageDataset, create_dataloaders
from model import load_checkpoint
from train import compute_binary_metrics, run_epoch, set_seed


def save_confusion_matrix_plot(metrics: dict[str, Any], output_dir: Path) -> Path:
    """Save a polished seaborn confusion-matrix plot."""

    matrix = np.asarray(metrics["confusion_matrix"])
    output_path = output_dir / "confusion_matrix.png"

    sns.set_theme(style="white")
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=True,
        linewidths=1.0,
        linecolor="white",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_roc_curve_plot(
    targets: list[int],
    positive_probabilities: list[float],
    metrics: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Save the ROC curve and annotate the AUC score."""

    positive_index = CLASS_TO_IDX["anemic"]
    output_path = output_dir / "roc_curve.png"
    figure, axis = plt.subplots(figsize=(7, 6))

    if len(set(targets)) >= 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            targets,
            positive_probabilities,
            pos_label=positive_index,
        )
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            color="#0f766e",
            linewidth=2.5,
            label=f"AUC = {metrics['auc_roc']:.4f}",
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Chance")
        axis.legend(loc="lower right")
    else:
        axis.text(
            0.5,
            0.5,
            "ROC curve is unavailable because only one class is present.",
            ha="center",
            va="center",
            wrap=True,
        )

    axis.set_title("ROC Curve", fontsize=14, fontweight="bold")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_precision_recall_curve_plot(
    targets: list[int],
    positive_probabilities: list[float],
    metrics: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Save the precision-recall curve and annotate average precision."""

    positive_index = CLASS_TO_IDX["anemic"]
    output_path = output_dir / "precision_recall_curve.png"

    figure, axis = plt.subplots(figsize=(7, 6))

    if len(set(targets)) >= 2:
        precision, recall, _ = precision_recall_curve(
            targets,
            positive_probabilities,
            pos_label=positive_index,
        )
        axis.plot(
            recall,
            precision,
            color="#b45309",
            linewidth=2.5,
            label=f"AP = {metrics['average_precision']:.4f}",
        )
        axis.legend(loc="lower left")
    else:
        axis.text(
            0.5,
            0.5,
            "Precision-recall curve is unavailable because only one class is present.",
            ha="center",
            va="center",
            wrap=True,
        )

    axis.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_per_class_metrics_table(metrics: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Save per-class metrics as both CSV and PNG table artifacts."""

    report = metrics["classification_report"]
    rows = []
    for row_name in ("non-anemic", "anemic", "macro avg", "weighted avg"):
        row = report[row_name]
        rows.append(
            {
                "class": row_name,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1-score": row["f1-score"],
                "support": row["support"],
            }
        )

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "per_class_metrics.csv"
    png_path = output_dir / "per_class_metrics_table.png"
    frame.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(9, 3.2))
    axis.axis("off")
    rounded_frame = frame.copy()
    rounded_frame["precision"] = rounded_frame["precision"].map(lambda value: f"{value:.4f}")
    rounded_frame["recall"] = rounded_frame["recall"].map(lambda value: f"{value:.4f}")
    rounded_frame["f1-score"] = rounded_frame["f1-score"].map(lambda value: f"{value:.4f}")
    rounded_frame["support"] = rounded_frame["support"].map(lambda value: f"{int(value)}")
    table = axis.table(
        cellText=rounded_frame.values,
        colLabels=rounded_frame.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    axis.set_title("Per-Class Metrics", fontsize=14, fontweight="bold", pad=16)
    figure.tight_layout()
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    return csv_path, png_path


def save_sample_predictions_grid(
    dataset: AnemiaImageDataset,
    predictions: list[int],
    probabilities: list[list[float]],
    output_dir: Path,
    max_samples: int = 12,
) -> Path:
    """Save a mixed grid of correct and incorrect predictions."""

    random.seed(TRAINING_CONFIG.random_seed)

    targets = dataset.targets
    correct_indices = [index for index, (target, prediction) in enumerate(zip(targets, predictions)) if target == prediction]
    incorrect_indices = [index for index, (target, prediction) in enumerate(zip(targets, predictions)) if target != prediction]

    chosen_correct = random.sample(correct_indices, k=min(len(correct_indices), max_samples // 2))
    chosen_incorrect = random.sample(incorrect_indices, k=min(len(incorrect_indices), max_samples // 2))
    selected_indices = chosen_correct + chosen_incorrect

    if len(selected_indices) < max_samples:
        remaining = [index for index in correct_indices if index not in selected_indices]
        selected_indices.extend(remaining[: max_samples - len(selected_indices)])

    if not selected_indices:
        raise RuntimeError("No predictions were available for the sample prediction grid.")

    columns = 4
    rows = int(np.ceil(len(selected_indices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.0 * rows))
    axes = np.atleast_2d(axes)

    for axis in axes.flat:
        axis.axis("off")

    positive_index = CLASS_TO_IDX["anemic"]

    for axis, sample_index in zip(axes.flat, selected_indices):
        sample = dataset.samples[sample_index]
        prediction = predictions[sample_index]
        probability_vector = probabilities[sample_index]
        confidence = probability_vector[prediction]
        is_correct = prediction == sample.label

        with Image.open(sample.path) as image:
            axis.imshow(image.convert("RGB"))

        axis.set_title(
            (
                f"True: {IDX_TO_CLASS[sample.label]}\n"
                f"Pred: {IDX_TO_CLASS[prediction]} | "
                f"Conf: {confidence:.2%}\n"
                f"P(anemic): {probability_vector[positive_index]:.2%}"
            ),
            fontsize=9,
        )

        edge_color = "#15803d" if is_correct else "#b91c1c"
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_edgecolor(edge_color)

    figure.suptitle("Sample Predictions (Correct + Incorrect)", fontsize=16, fontweight="bold")
    figure.tight_layout()
    output_path = output_dir / "sample_predictions_grid.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def evaluate(checkpoint_path: Path, dataset_root: Path, split: str = "test") -> dict[str, Any]:
    """Run evaluation on the requested split and generate all Stage 2 artifacts."""

    set_seed(TRAINING_CONFIG.random_seed)
    output_dir = EVALUATION_LOGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    data_bundle = create_dataloaders(
        dataset_root=dataset_root,
        batch_size=TRAINING_CONFIG.batch_size,
        show_statistics=True,
    )
    device = TRAINING_CONFIG.device
    model, checkpoint = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    criterion = torch.nn.CrossEntropyLoss()

    split_loader = data_bundle.test_loader if split == "test" else data_bundle.valid_loader
    split_dataset = data_bundle.test_dataset if split == "test" else data_bundle.valid_dataset

    result = run_epoch(
        model=model,
        data_loader=split_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        description=f"Evaluate {split}",
    )
    metrics = compute_binary_metrics(
        result.targets,
        result.predictions,
        result.probabilities,
    )

    save_confusion_matrix_plot(metrics, output_dir)
    save_roc_curve_plot(result.targets, metrics["positive_probabilities"], metrics, output_dir)
    save_precision_recall_curve_plot(
        result.targets,
        metrics["positive_probabilities"],
        metrics,
        output_dir,
    )
    save_per_class_metrics_table(metrics, output_dir)
    save_sample_predictions_grid(
        dataset=split_dataset,
        predictions=result.predictions,
        probabilities=result.probabilities,
        output_dir=output_dir,
    )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": split,
        "loss": result.loss,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "auc_roc": metrics["auc_roc"],
        "average_precision": metrics["average_precision"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "classification_report": metrics["classification_report"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    summary_path = output_dir / f"{split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nEvaluation summary")
    print("-" * 72)
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Epoch:        {checkpoint.get('epoch')}")
    print(f"Split:        {split}")
    print(f"Loss:         {result.loss:.4f}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"F1:           {metrics['f1']:.4f}")
    print(
        f"AUC-ROC:      {metrics['auc_roc']:.4f}"
        if np.isfinite(metrics["auc_roc"])
        else "AUC-ROC:      nan"
    )
    print(f"Sensitivity:  {metrics['sensitivity']:.4f}")
    print(f"Specificity:  {metrics['specificity']:.4f}")
    print("-" * 72)
    print(metrics["classification_report_text"])

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate the best anemia detection model.")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_CHECKPOINT_PATH))
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    evaluate(
        checkpoint_path=Path(arguments.checkpoint).resolve(),
        dataset_root=Path(arguments.dataset_root).resolve(),
        split=arguments.split,
    )
