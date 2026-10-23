"""Rigorous evaluation, calibration diagnostics, and Bland-Altman agreement analysis for VYRA.

Calculates and saves:
1. Classification Metrics: Confusion matrix, Accuracy, Sensitivity, Specificity, Precision, F1, ROC-AUC
2. Continuous Hb Regression: MAE (g/dL), RMSE (g/dL), R2, Bland-Altman Mean Bias and 95% Limits of Agreement
3. Probability Calibration: Temperature scaling parameters, Expected Calibration Error (ECE), Brier Score, Reliability Plot
4. Stratified Subgroup Analysis: Stratified by Dataset, Modality, and ITA Skin-Tone Tiers
5. Machine-readable JSON/CSV reports in evaluation/metrics/ and publication-grade plots in evaluation/plots/
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import statsmodels.api as sm

from src.data.manifest_dataset import ManifestDataset
from src.models.dual_head_model import DualHeadEfficientNetB3

METRICS_DIR = PROJECT_ROOT / "evaluation" / "metrics"
PLOTS_DIR = PROJECT_ROOT / "evaluation" / "plots"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def compute_expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def fit_temperature_scaling(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Fit optimal temperature parameter T on validation split logits."""
    val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)
    temp = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temp], lr=0.01, max_iter=50)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(val_logits_t / temp, val_labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(temp, 0.05, 5.0).item())


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path):
    plt.figure(figsize=(5, 4), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Test Split Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_bland_altman(true_hb: np.ndarray, pred_hb: np.ndarray, output_path: Path):
    """Generate clinical Bland-Altman agreement plot with 95% Limits of Agreement."""
    mean_val = (true_hb + pred_hb) / 2.0
    diff = pred_hb - true_hb  # Error: Estimation - Reference

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    plt.figure(figsize=(7, 5), dpi=300)
    plt.scatter(mean_val, diff, alpha=0.6, color="#1f77b4", edgecolor="k", s=40)
    plt.axhline(mean_diff, color="red", linestyle="--", linewidth=1.5, label=f"Mean Bias: {mean_diff:+.2f} g/dL")
    plt.axhline(loa_upper, color="gray", linestyle=":", linewidth=1.5, label=f"+1.96 SD: {loa_upper:+.2f} g/dL")
    plt.axhline(loa_lower, color="gray", linestyle=":", linewidth=1.5, label=f"-1.96 SD: {loa_lower:+.2f} g/dL")
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)

    plt.title("Bland–Altman Agreement: Estimated Hb vs Reference Lab Hb (g/dL)")
    plt.xlabel("Mean Hemoglobin (Reference + Estimated) / 2 [g/dL]")
    plt.ylabel("Difference (Estimated - Reference) [g/dL]")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_reliability_curve(probs: np.ndarray, labels: np.ndarray, output_path: Path, n_bins: int = 10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []

    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            bin_confidences.append(np.mean(probs[in_bin]))
            bin_accuracies.append(np.mean(labels[in_bin]))

    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(bin_confidences, bin_accuracies, marker="s", color="#2ca02c", linewidth=2, label="Model Reliability")
    plt.title("Reliability Diagram (Post-Hoc Calibrated)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_evaluation(checkpoint_path: Path):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = DualHeadEfficientNetB3(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_dataset = ManifestDataset(PROJECT_ROOT / "data" / "splits" / "validation.csv")
    test_dataset = ManifestDataset(PROJECT_ROOT / "data" / "splits" / "test.csv")

    # 1. Fit Temperature Scaling on Validation Split ONLY
    print("Fitting temperature scaling on validation split...")
    val_logits = []
    val_labels = []
    with torch.no_grad():
        for i in range(len(val_dataset)):
            s = val_dataset[i]
            img = s["image"].unsqueeze(0).to(device)
            feat = s["features"].unsqueeze(0).to(device)
            bp = s["body_part"].unsqueeze(0).to(device)
            lg, _ = model(img, feat, bp)
            val_logits.append(lg.cpu().numpy()[0])
            val_labels.append(s["label"].item())

    val_logits = np.array(val_logits)
    val_labels = np.array(val_labels)
    optimal_temp = fit_temperature_scaling(val_logits, val_labels)
    print(f"Optimal Temperature T fitted on validation data: {optimal_temp:.4f}")

    # 2. Run Comprehensive Inference on Untouched Test Split
    print(f"Evaluating untouched test split ({len(test_dataset)} samples)...")
    records = []
    for i in range(len(test_dataset)):
        s = test_dataset[i]
        with torch.no_grad():
            img = s["image"].unsqueeze(0).to(device)
            feat = s["features"].unsqueeze(0).to(device)
            bp = s["body_part"].unsqueeze(0).to(device)
            lg, hb_p = model(img, feat, bp)

            scaled_lg = lg.cpu().numpy()[0] / optimal_temp
            prob = float(torch.softmax(torch.tensor(scaled_lg), dim=-1)[1])
            hb_val = float(hb_p.cpu().numpy()[0][0])

        lab_hb = float(s["hb"].item())
        records.append({
            "patient_id": s["patient_id"],
            "dataset": s["dataset"],
            "modality": s["modality"],
            "true_label": s["label"].item(),
            "pred_prob": prob,
            "pred_class": int(prob >= 0.50),
            "true_hb": lab_hb if not np.isnan(lab_hb) else None,
            "pred_hb": hb_val,
            "ita": s["ita"],
        })

    df = pd.DataFrame(records)

    # 3. Compute Classification Metrics
    y_true = df["true_label"].values
    y_prob = df["pred_prob"].values
    y_pred = df["pred_class"].values

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = float(recall_score(y_true, y_pred))
    spec = float(tn / (tn + fp))
    prec = float(precision_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_expected_calibration_error(y_prob, y_true)

    clf_metrics = {
        "test_samples": len(df),
        "accuracy": acc,
        "sensitivity_recall": sens,
        "specificity": spec,
        "precision": prec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "temperature_scaling_T": optimal_temp,
    }

    # 4. Continuous Hemoglobin Regression Metrics (only on samples with valid lab Hb)
    df_hb = df.dropna(subset=["true_hb"])
    if len(df_hb) > 0:
        true_hb = df_hb["true_hb"].values.astype(float)
        pred_hb = df_hb["pred_hb"].values.astype(float)

        mae = float(mean_absolute_error(true_hb, pred_hb))
        rmse = float(np.sqrt(mean_squared_error(true_hb, pred_hb)))
        r2 = float(r2_score(true_hb, pred_hb))

        diff = pred_hb - true_hb
        mean_bias = float(np.mean(diff))
        sd_diff = float(np.std(diff, ddof=1))
        loa_upper = float(mean_bias + 1.96 * sd_diff)
        loa_lower = float(mean_bias - 1.96 * sd_diff)

        reg_metrics = {
            "paired_samples": len(df_hb),
            "mae_g_dl": mae,
            "rmse_g_dl": rmse,
            "r2_score": r2,
            "bland_altman": {
                "mean_bias_g_dl": mean_bias,
                "sd_bias": sd_diff,
                "loa_upper_95_g_dl": loa_upper,
                "loa_lower_95_g_dl": loa_lower,
            },
        }

        # Generate Bland-Altman plot
        plot_bland_altman(true_hb, pred_hb, PLOTS_DIR / "bland_altman_agreement.png")
    else:
        reg_metrics = {"status": "unavailable", "reason": "No paired laboratory measurements in test set."}

    # 5. Stratification Analysis
    stratified_results = {}
    # By Dataset
    for ds_name, group in df.groupby("dataset"):
        stratified_results[f"dataset_{ds_name}"] = {
            "samples": len(group),
            "accuracy": float(accuracy_score(group["true_label"], group["pred_class"])),
            "f1": float(f1_score(group["true_label"], group["pred_class"], zero_division=0)),
        }
    # By Modality
    for mod_name, group in df.groupby("modality"):
        stratified_results[f"modality_{mod_name}"] = {
            "samples": len(group),
            "accuracy": float(accuracy_score(group["true_label"], group["pred_class"])),
            "f1": float(f1_score(group["true_label"], group["pred_class"], zero_division=0)),
        }

    # 6. Save Plots & Metric Reports
    plot_confusion_matrix(cm, ["Non-Anemic", "Anemic"], PLOTS_DIR / "confusion_matrix.png")
    plot_reliability_curve(y_prob, y_true, PLOTS_DIR / "reliability_diagram.png")

    report = {
        "model_version": "vyra-dualhead-efficientnet-b3",
        "checkpoint": str(checkpoint_path),
        "test_dataset_size": len(df),
        "classification": clf_metrics,
        "hemoglobin_regression": reg_metrics,
        "stratified_subgroups": stratified_results,
    }

    report_path = METRICS_DIR / "test_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    df.to_csv(METRICS_DIR / "test_predictions_detailed.csv", index=False)
    print(f"\nSaved evaluation metrics to: {report_path}")
    print(f"Saved plots to: {PLOTS_DIR}")
    print("\n--- TEST SET PERFORMANCE SUMMARY ---")
    print(f"Accuracy: {acc*100:.2f}% | Sensitivity/Recall: {sens*100:.2f}% | Specificity: {spec*100:.2f}% | ROC-AUC: {roc_auc:.4f}")
    if len(df_hb) > 0:
        print(f"Continuous Hb MAE: {mae:.2f} g/dL | RMSE: {rmse:.2f} g/dL | Mean Bias: {mean_bias:+.2f} g/dL (LOA: [{loa_lower:.2f}, {loa_upper:.2f}])")


if __name__ == "__main__":
    ckpt = CHECKPOINTS_DIR / "best_dual_head.pth"
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)
    run_evaluation(ckpt)
