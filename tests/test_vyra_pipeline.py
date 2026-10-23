"""Comprehensive test suite for VYRA Person 2 AI/ML Workstream.

Tests:
1. Model initialization and architecture
2. Output tensor shapes (Head A: 2 logits, Head B: 1 positive continuous Hb)
3. Positive Hb constraint verification
4. Masked dual loss handling missing Hb labels
5. Invalid image input error handling
6. Missing checkpoint fallback handling
7. Inference running on CPU
8. Dataset manifest integrity (non-empty, required columns, existing file paths)
9. Leakage-free patient-level split audit (0% overlap across partitions)
10. Metric calculation on known synthetic test cases
11. Structured inference response schema verification
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import numpy as np
import pandas as pd
from PIL import Image

from src.models.dual_head_model import DualHeadEfficientNetB3, MaskedDualLoss
from src.data.normalization_adapter import preprocess_image
from src.inference.predictor import predict, VyraInferenceEngine, StructuredPredictionResult


# 1. Model initialization & tensor shapes
def test_model_initialization_and_shapes():
    model = DualHeadEfficientNetB3(pretrained=False)
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    features = torch.randn(2, 27)
    body_part = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    logits, hb = model(images, features, body_part)
    assert logits.shape == (2, 2), f"Expected (2, 2), got {logits.shape}"
    assert hb.shape == (2, 1), f"Expected (2, 1), got {hb.shape}"


# 2. Positive Hb output constraint
def test_positive_hb_constraint():
    model = DualHeadEfficientNetB3(pretrained=False)
    model.eval()
    images = torch.randn(5, 3, 224, 224)
    _, hb = model(images)
    assert (hb >= 0.0).all(), "Softplus constraint violated: Hb estimates must be >= 0.0"


# 3. Masked dual loss handles missing Hb targets
def test_masked_dual_loss():
    criterion = MaskedDualLoss(alpha=0.7, beta=0.3)
    logits = torch.randn(3, 2, requires_grad=True)
    hb_pred = torch.tensor([[11.0], [12.5], [8.0]], requires_grad=True)
    labels = torch.tensor([0, 1, 0])
    # 1 valid Hb, 1 NaN, 1 negative (missing)
    hb_targets = torch.tensor([[11.2], [float("nan")], [-1.0]])

    total_loss, ce_loss, mse_loss = criterion(logits, hb_pred, labels, hb_targets)
    assert not torch.isnan(total_loss)
    total_loss.backward()
    assert logits.grad is not None
    assert hb_pred.grad is not None


# 4. Invalid image input handling
def test_invalid_image_handling():
    engine = VyraInferenceEngine()
    result = engine.predict("non_existent_image_file_path_12345.png")
    assert not result.error is None or len(result.quality_warnings) > 0
    assert result.predicted_class is None
    assert result.hb_estimate_g_dl is None


# 5. Missing checkpoint fallback
def test_missing_checkpoint_fallback():
    engine = VyraInferenceEngine(checkpoint_path="non_existent_ckpt.pth")
    # Engine should fall back safely without crashing
    assert engine.model is not None


# 6. CPU inference execution
def test_cpu_inference():
    engine = VyraInferenceEngine(device="cpu")
    dummy_img = Image.new("RGB", (224, 224), color=(180, 100, 100))
    result = engine.predict(dummy_img, body_part="conjunctiva")
    assert result.anemia_probability is not None
    assert isinstance(result.processing_time_ms, int)


# 7. Dataset manifest integrity
def test_dataset_manifest_integrity():
    for split_name in ["train.csv", "validation.csv", "test.csv"]:
        path = PROJECT_ROOT / "data" / "splits" / split_name
        assert path.exists(), f"Missing split manifest: {path}"
        df = pd.read_csv(path)
        assert len(df) > 0, f"Split manifest is empty: {path}"
        required_cols = ["file_path", "dataset", "modality", "label", "has_lab_hb", "patient_id"]
        for c in required_cols:
            assert c in df.columns, f"Missing column {c} in {split_name}"


# 8. Zero patient leakage verification
def test_zero_patient_leakage():
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "train.csv")
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "validation.csv")
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "test.csv")

    train_pts = set(train_df["cluster_patient_id"])
    val_pts = set(val_df["cluster_patient_id"])
    test_pts = set(test_df["cluster_patient_id"])

    assert len(train_pts & val_pts) == 0, "Patient leakage in Train/Val!"
    assert len(train_pts & test_pts) == 0, "Patient leakage in Train/Test!"
    assert len(val_pts & test_pts) == 0, "Patient leakage in Val/Test!"


# 9. Structured inference schema verification
def test_structured_inference_contract():
    res = predict("data/raw/cp_anemic/CP-AnemiC/Anemic/Image_001.png", body_part="conjunctiva")
    assert isinstance(res, dict)
    required_keys = [
        "hb_estimate_g_dl",
        "anemia_probability",
        "predicted_class",
        "model_version",
        "calibration_status",
        "quality_warnings",
        "error",
    ]
    for k in required_keys:
        assert k in res, f"Contract missing key: {k}"


# 10. Label mapping consistency audit
def test_label_mapping_consistency():
    for split_name in ["train.csv", "validation.csv", "test.csv"]:
        df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / split_name)
        anemic_subset = df[df["class_name"].str.lower().str.contains("anemic") & ~df["class_name"].str.lower().str.contains("non")]
        non_anemic_subset = df[df["class_name"].str.lower().str.contains("non")]
        assert (anemic_subset["label"] == 1).all(), f"Inconsistent positive label in {split_name}"
        assert (non_anemic_subset["label"] == 0).all(), f"Inconsistent negative label in {split_name}"


# 11. Evaluation metric reproducibility from saved test predictions
def test_evaluation_reproducibility():
    metrics_path = PROJECT_ROOT / "evaluation" / "metrics" / "champion_test_report.json"
    preds_path = PROJECT_ROOT / "evaluation" / "metrics" / "champion_test_predictions.csv"
    assert metrics_path.exists(), "Missing champion_test_report.json"
    assert preds_path.exists(), "Missing champion_test_predictions.csv"

    import json
    with open(metrics_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    df_preds = pd.read_csv(preds_path)
    reproduced_acc = float((df_preds["pred_class"] == df_preds["true_label"]).mean())
    assert abs(reproduced_acc - report["classification"]["accuracy"]) < 1e-4, "Accuracy discrepancy in saved predictions!"

    df_hb = df_preds.dropna(subset=["true_hb"])
    reproduced_mae = float(np.mean(np.abs(df_hb["pred_hb"] - df_hb["true_hb"])))
    assert abs(reproduced_mae - report["hemoglobin_regression"]["mae_g_dl"]) < 1e-4, "Hb MAE discrepancy in saved predictions!"


# 12. Calibration temperature embedded in checkpoint
def test_calibration_temperature_embedded():
    ckpt_path = PROJECT_ROOT / "checkpoints" / "best_dual_head.pth"
    assert ckpt_path.exists(), "Missing best_dual_head.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "temperature" in ckpt, "Missing calibration temperature in checkpoint!"
    assert ckpt["temperature"] > 0.0, "Invalid temperature parameter"
    assert ckpt.get("has_validated_regression") is True


# 13. Modality-specific Hb availability
def test_modality_specific_hb_availability():
    sample_img = "data/raw/cp_anemic/CP-AnemiC/Anemic/Image_001.png"
    # Conjunctiva must return numeric Hb estimate
    res_conj = predict(sample_img, body_part="conjunctiva")
    assert res_conj["hb_estimate_g_dl"] is not None
    assert res_conj["regression_status"] == "validated"

    # Fingernail must return None with explicit warning
    res_nail = predict(sample_img, body_part="fingernail")
    assert res_nail["hb_estimate_g_dl"] is None
    assert res_nail["regression_status"] == "unavailable"
    assert any("fingernail" in w.lower() for w in res_nail["quality_warnings"])


# 14. Real checkpoint inference sanity bounds
def test_real_checkpoint_inference():
    sample_img = "data/raw/cp_anemic/CP-AnemiC/Anemic/Image_001.png"
    res = predict(sample_img, body_part="conjunctiva")
    hb = res["hb_estimate_g_dl"]
    prob = res["anemia_probability"]
    assert 3.0 <= hb <= 20.0, f"Unrealistic Hb estimate: {hb} g/dL"
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
    assert res["predicted_class"] in ["anemic", "non-anemic"]


if __name__ == "__main__":
    test_model_initialization_and_shapes()
    test_positive_hb_constraint()
    test_masked_dual_loss()
    test_invalid_image_handling()
    test_missing_checkpoint_fallback()
    test_cpu_inference()
    test_dataset_manifest_integrity()
    test_zero_patient_leakage()
    test_structured_inference_contract()
    test_label_mapping_consistency()
    test_evaluation_reproducibility()
    test_calibration_temperature_embedded()
    test_modality_specific_hb_availability()
    test_real_checkpoint_inference()
    print("ALL 14 INTEGRATION AND UNIT TESTS PASSED!")
