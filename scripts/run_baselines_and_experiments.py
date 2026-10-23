"""Methodical Experimentation Suite for VYRA Person 2.

Runs and logs:
1. Baseline 1: Classification-Only (alpha=1.0, beta=0.0)
2. Baseline 2: Regression-Only on valid paired conjunctiva data (alpha=0.0, beta=1.0)
3. Champion: Joint Dual-Head with clinical mean bias initialization and Smooth L1 loss (alpha=0.7, beta=0.3)

Strict Evaluation Protocol:
- All model selection and early stopping use VALIDATION data only.
- Temperature scaling is fitted on VALIDATION data only.
- The held-out TEST set is evaluated once for final reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.manifest_dataset import ManifestDataset, create_manifest_dataloaders
from src.models.dual_head_model import DualHeadEfficientNetB3, MaskedDualLoss

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
METRICS_DIR = PROJECT_ROOT / "evaluation" / "metrics"
PLOTS_DIR = PROJECT_ROOT / "evaluation" / "plots"


def compute_class_weights(train_csv: Path) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    labels = df["label"].values
    counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(counts) * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, regression_only=False):
    model.train()
    total_loss, total_ce, total_reg = 0.0, 0.0, 0.0
    correct, total_samples = 0, 0
    hb_abs_errors = []

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        body_part = batch["body_part"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        hb = batch["hb"].to(device, non_blocking=True)

        if images.size(0) <= 1:
            continue

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits, hb_pred = model(images, features, body_part)
            loss, ce, reg = criterion(logits, hb_pred, labels, hb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_ce += ce.item() * bs
        total_reg += reg.item() * bs
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += bs

        valid_hb = ~torch.isnan(hb) & (hb > 0.0)
        if valid_hb.sum() > 0:
            diffs = torch.abs(hb_pred[valid_hb] - hb[valid_hb]).detach().cpu().numpy()
            hb_abs_errors.extend(diffs.flatten())

    mae = float(np.mean(hb_abs_errors)) if hb_abs_errors else 0.0
    acc = correct / total_samples if total_samples > 0 else 0.0
    return {
        "loss": total_loss / max(total_samples, 1),
        "ce_loss": total_ce / max(total_samples, 1),
        "reg_loss": total_reg / max(total_samples, 1),
        "accuracy": acc,
        "hb_mae": mae,
    }


def evaluate_metrics(model, loader, criterion, device):
    model.eval()
    total_loss, total_ce, total_reg = 0.0, 0.0, 0.0
    correct, total_samples = 0, 0
    hb_abs_errors = []
    hb_sq_errors = []
    y_true, y_prob = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            features = batch["features"].to(device, non_blocking=True)
            body_part = batch["body_part"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            hb = batch["hb"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                logits, hb_pred = model(images, features, body_part)
                loss, ce, reg = criterion(logits, hb_pred, labels, hb)

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_ce += ce.item() * bs
            total_reg += reg.item() * bs
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += bs

            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            y_prob.extend(probs)
            y_true.extend(labels.cpu().numpy())

            valid_hb = ~torch.isnan(hb) & (hb > 0.0)
            if valid_hb.sum() > 0:
                diffs = torch.abs(hb_pred[valid_hb] - hb[valid_hb]).detach().cpu().numpy().flatten()
                hb_abs_errors.extend(diffs)
                hb_sq_errors.extend(diffs ** 2)

    mae = float(np.mean(hb_abs_errors)) if hb_abs_errors else 0.0
    rmse = float(np.sqrt(np.mean(hb_sq_errors))) if hb_sq_errors else 0.0
    acc = correct / max(total_samples, 1)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    return {
        "loss": total_loss / max(total_samples, 1),
        "ce_loss": total_ce / max(total_samples, 1),
        "reg_loss": total_reg / max(total_samples, 1),
        "accuracy": acc,
        "auc": auc,
        "hb_mae": mae,
        "hb_rmse": rmse,
    }


def fit_temperature(model, val_dataset, device):
    """Fit optimal temperature on validation logits."""
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for i in range(len(val_dataset)):
            s = val_dataset[i]
            img = s["image"].unsqueeze(0).to(device)
            feat = s["features"].unsqueeze(0).to(device)
            bp = s["body_part"].unsqueeze(0).to(device)
            lg, _ = model(img, feat, bp)
            logits_list.append(lg.cpu().numpy()[0])
            labels_list.append(s["label"].item())

    val_logits_t = torch.tensor(np.array(logits_list), dtype=torch.float32)
    val_labels_t = torch.tensor(np.array(labels_list), dtype=torch.long)
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


def run_experiment(
    exp_name: str,
    alpha: float,
    beta: float,
    init_bias: bool = True,
    regression_loss_type: str = "smooth_l1",
    regression_only: bool = False,
    epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, Any]:
    print(f"\n{'='*65}\nRUNNING EXPERIMENT: {exp_name}\n{'='*65}")
    ckpt_path = CHECKPOINTS_DIR / f"{exp_name}.pth"
    if ckpt_path.exists():
        try:
            cached = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "history" in cached and len(cached["history"]) == epochs:
                print(f"Loaded existing finished experiment checkpoint: {ckpt_path}")
                return {
                    "exp_name": exp_name,
                    "checkpoint": str(ckpt_path),
                    "best_val_metrics": cached["val_metrics"],
                    "history": cached["history"],
                }
        except Exception:
            pass

    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "splits" / "validation.csv"

    if regression_only:
        # Create paired-only manifests for pure regression training
        df_tr = pd.read_csv(train_csv)
        tr_hb_num = pd.to_numeric(df_tr["lab_hb_g_dl"], errors="coerce")
        df_tr_paired = df_tr[tr_hb_num.notna() & (tr_hb_num > 0.0)]
        paired_tr_csv = PROJECT_ROOT / "data" / "splits" / "train_paired_conjunctiva.csv"
        df_tr_paired.to_csv(paired_tr_csv, index=False)

        df_vl = pd.read_csv(val_csv)
        vl_hb_num = pd.to_numeric(df_vl["lab_hb_g_dl"], errors="coerce")
        df_vl_paired = df_vl[vl_hb_num.notna() & (vl_hb_num > 0.0)]
        paired_vl_csv = PROJECT_ROOT / "data" / "splits" / "val_paired_conjunctiva.csv"
        df_vl_paired.to_csv(paired_vl_csv, index=False)

        active_train_csv = paired_tr_csv
        active_val_csv = paired_vl_csv
    else:
        active_train_csv = train_csv
        active_val_csv = val_csv

    class_weights = compute_class_weights(active_train_csv).to(device) if alpha > 0 else None

    loaders = create_manifest_dataloaders(
        train_manifest=active_train_csv,
        val_manifest=active_val_csv,
        batch_size=batch_size,
        num_workers=0,
    )

    model = DualHeadEfficientNetB3(pretrained=True).to(device)
    if init_bias:
        model.init_regression_head(initial_hb_mean=10.75)

    criterion = MaskedDualLoss(
        alpha=alpha,
        beta=beta,
        class_weights=class_weights,
        regression_loss_type=regression_loss_type,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_score = float("inf")
    best_weights = None
    best_metrics = {}
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device, regression_only=regression_only)
        vl = evaluate_metrics(model, loaders["val"], criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{ep:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {tr['loss']:.4f} (Acc: {tr['accuracy']*100:.1f}%, MAE: {tr['hb_mae']:.2f}) | "
            f"Val Loss: {vl['loss']:.4f} (Acc: {vl['accuracy']*100:.1f}%, AUC: {vl['auc']:.4f}, MAE: {vl['hb_mae']:.2f})"
        )
        history.append({"epoch": ep, "train": tr, "val": vl, "time": elapsed})

        # Selection criterion: primary validation loss
        val_score = vl["loss"]
        if val_score < best_val_score:
            best_val_score = val_score
            best_metrics = vl
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    ckpt_path = CHECKPOINTS_DIR / f"{exp_name}.pth"
    torch.save({
        "exp_name": exp_name,
        "model_state_dict": best_weights,
        "val_metrics": best_metrics,
        "alpha": alpha,
        "beta": beta,
        "history": history,
    }, ckpt_path)
    print(f"Saved {exp_name} checkpoint to {ckpt_path} (Best Val Loss: {best_val_score:.4f})")

    return {
        "exp_name": exp_name,
        "checkpoint": str(ckpt_path),
        "best_val_metrics": best_metrics,
        "history": history,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}

    # 1. Baseline 1: Classification-Only (alpha=1.0, beta=0.0)
    results["baseline1_clf_only"] = run_experiment(
        exp_name="baseline1_clf_only",
        alpha=1.0, beta=0.0,
        init_bias=False,
        epochs=5,
        device=device,
    )

    # 2. Baseline 2: Regression-Only on valid paired data (alpha=0.0, beta=1.0)
    results["baseline2_reg_only"] = run_experiment(
        exp_name="baseline2_reg_only",
        alpha=0.0, beta=1.0,
        init_bias=True,
        regression_loss_type="smooth_l1",
        regression_only=True,
        epochs=5,
        device=device,
    )

    # 3. Champion: Joint Dual-Head with clinical bias initialization & Smooth L1 loss
    results["champion_joint_dualhead"] = run_experiment(
        exp_name="champion_joint_dualhead",
        alpha=0.70, beta=0.30,
        init_bias=True,
        regression_loss_type="smooth_l1",
        epochs=5,
        device=device,
    )

    # Save experiment comparison report
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    comp_path = LOGS_DIR / "baselines_and_experiments_summary.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPARISON SUMMARY (VALIDATION SPLIT)")
    print("=" * 65)
    for name, res in results.items():
        vm = res["best_val_metrics"]
        print(f"[{name}]: Val Loss={vm['loss']:.4f}, Val Acc={vm['accuracy']*100:.2f}%, Val AUC={vm['auc']:.4f}, Val Hb MAE={vm['hb_mae']:.2f} g/dL, Val Hb RMSE={vm['hb_rmse']:.2f} g/dL")

    # Evaluate the Champion Model on Held-Out Test Set
    champ_ckpt = CHECKPOINTS_DIR / "champion_joint_dualhead.pth"
    print(f"\nEvaluating Champion Model on Untouched Test Set ({champ_ckpt})...")

    # Load champion weights
    ckpt = torch.load(champ_ckpt, map_location=device, weights_only=False)
    champ_model = DualHeadEfficientNetB3(pretrained=False).to(device)
    champ_model.load_state_dict(ckpt["model_state_dict"])
    champ_model.eval()

    val_ds = ManifestDataset(PROJECT_ROOT / "data" / "splits" / "validation.csv")
    test_ds = ManifestDataset(PROJECT_ROOT / "data" / "splits" / "test.csv")

    # Fit calibration on validation data ONLY
    opt_temp = fit_temperature(champ_model, val_ds, device)
    print(f"Champion Optimal Temperature T (fitted on validation data): {opt_temp:.4f}")

    # Embed temperature into checkpoint
    ckpt["temperature"] = opt_temp
    ckpt["has_validated_regression"] = True
    torch.save(ckpt, champ_ckpt)
    # Also save as best_dual_head.pth for production inference
    torch.save(ckpt, CHECKPOINTS_DIR / "best_dual_head.pth")

    # Test inference
    test_records = []
    for i in range(len(test_ds)):
        s = test_ds[i]
        with torch.no_grad():
            img = s["image"].unsqueeze(0).to(device)
            feat = s["features"].unsqueeze(0).to(device)
            bp = s["body_part"].unsqueeze(0).to(device)
            lg, hb_p = champ_model(img, feat, bp)

            scaled_lg = lg.cpu().numpy()[0] / opt_temp
            prob = float(torch.softmax(torch.tensor(scaled_lg), dim=-1)[1])
            hb_val = float(hb_p.cpu().numpy()[0][0])

        lab_hb = float(s["hb"].item())
        test_records.append({
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

    df_test = pd.DataFrame(test_records)
    y_true = df_test["true_label"].values
    y_prob = df_test["pred_prob"].values
    y_pred = df_test["pred_class"].values

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = float(recall_score(y_true, y_pred))
    spec = float(tn / (tn + fp))
    prec = float(precision_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # Continuous Hb metrics
    df_hb = df_test.dropna(subset=["true_hb"])
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

    # Save champion test predictions & report
    df_test.to_csv(METRICS_DIR / "champion_test_predictions.csv", index=False)
    champ_report = {
        "model_version": "vyra-champion-dualhead-efficientnet-b3",
        "checkpoint": str(champ_ckpt),
        "temperature_scaling_T": opt_temp,
        "test_dataset_size": len(df_test),
        "classification": {
            "accuracy": acc,
            "sensitivity_recall": sens,
            "specificity": spec,
            "precision": prec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        },
        "hemoglobin_regression": {
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
        },
    }
    with open(METRICS_DIR / "champion_test_report.json", "w", encoding="utf-8") as f:
        json.dump(champ_report, f, indent=2)

    # Generate publication-grade plots
    # 1. Confusion Matrix
    plt.figure(figsize=(5, 4), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Anemic", "Anemic"], yticklabels=["Non-Anemic", "Anemic"])
    plt.title("Champion Model Test Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "champion_confusion_matrix.png")
    plt.close()

    # 2. Bland-Altman Plot
    mean_val = (true_hb + pred_hb) / 2.0
    plt.figure(figsize=(7, 5), dpi=300)
    plt.scatter(mean_val, diff, alpha=0.6, color="#1f77b4", edgecolor="k", s=40)
    plt.axhline(mean_bias, color="red", linestyle="--", linewidth=1.5, label=f"Mean Bias: {mean_bias:+.2f} g/dL")
    plt.axhline(loa_upper, color="gray", linestyle=":", linewidth=1.5, label=f"+1.96 SD: {loa_upper:+.2f} g/dL")
    plt.axhline(loa_lower, color="gray", linestyle=":", linewidth=1.5, label=f"-1.96 SD: {loa_lower:+.2f} g/dL")
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    plt.title("Champion Bland–Altman Agreement: Estimated Hb vs Reference Lab CBC (g/dL)")
    plt.xlabel("Mean Hemoglobin (Reference + Estimated) / 2 [g/dL]")
    plt.ylabel("Difference (Estimated - Reference) [g/dL]")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "champion_bland_altman.png")
    plt.close()

    print("\n" + "=" * 65)
    print("CHAMPION MODEL HELD-OUT TEST PERFORMANCE")
    print("=" * 65)
    print(f"Accuracy: {acc*100:.2f}% | Sensitivity: {sens*100:.2f}% | Specificity: {spec*100:.2f}% | ROC-AUC: {roc_auc:.4f}")
    print(f"Continuous Hb MAE: {mae:.2f} g/dL | RMSE: {rmse:.2f} g/dL | R2: {r2:.4f}")
    print(f"Bland-Altman Mean Bias: {mean_bias:+.2f} g/dL | 95% LOA: [{loa_lower:.2f}, {loa_upper:.2f}] g/dL")
    print(f"Saved Champion report to {METRICS_DIR / 'champion_test_report.json'}")


if __name__ == "__main__":
    main()
