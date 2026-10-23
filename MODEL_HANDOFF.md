# VYRA — Production Model & Weights Handoff Guide

This document accompanies the production model package (`vyra-model-handoff.zip`) prepared for connecting the frontend and external services to the VYRA dual-head screening architecture.

---

## 1. Package Inventory & Destination Mapping

Extract the contents of `vyra-model-handoff.zip` into your cloned repository root (`Vyra---GCEM-`):

| File in Zip Package | Destination in Cloned Repo | Description & Purpose |
| :--- | :--- | :--- |
| `checkpoints/best_dual_head.pth` | `checkpoints/best_dual_head.pth` *(and mirror to `models/best_dual_head.pth`)* | **Production Champion Checkpoint**. Full PyTorch weights for `DualHeadEfficientNetB3` (classification + regression) with embedded calibration and regression metadata. |
| `models/temperature_scaling.json` | `models/temperature_scaling.json` | **Standalone Temperature Config**. Contains `T = 1.3391530513763428` matching the champion evaluation report. |
| `requirements-tested.txt` | `requirements-tested.txt` *(repo root)* | **Harmonized Dependency File**. Resolves constraints between Person 1 (`person1/pyproject.toml`) and Person 3 backend. |
| `samples/test_eye_conjunctiva.png` | `samples/test_eye_conjunctiva.png` | **Permitted Conjunctiva Test Sample** from held-out test split (`CP-AnemiC V2 / Image_002.png`, lab Hb: 9.9 g/dL). |
| `samples/test_fingernail.png` | `samples/test_fingernail.png` | **Permitted Fingernail Test Sample** from held-out test split (`Ghana Fingernails / Anemic-Fin-007 (10).png`). |
| `samples/sample_predictions_reference.json` | `samples/sample_predictions_reference.json` | **Unedited Reference Inference Outputs** produced by running the model on the two sample test images. |
| `MODEL_HANDOFF.md` | `MODEL_HANDOFF.md` *(repo root)* | This documentation and verification manifest. |

---

## 2. Model Version & Checksum Reconciliation

### Checkpoint Metadata
- **Architecture**: `DualHeadEfficientNetB3` (Torchvision EfficientNet-B3 backbone with custom dual-head: binary classification + continuous Hb regression).
- **Exact File Size**: `47,314,074` bytes (~45.1 MB).
- **Actual SHA-256 Checksum**:
  ```text
  57b6a4be8b14d11c2574c64192983942ae3d6ba760ddc3fca0828341f5d28157
  ```

### Reconciling Documented Checksum Differences
- `docs/PERSON_2_FINAL_HANDOFF.md` (Line 94) records:
  `57b6a4be8b14d11c2574c64192983942ae3d6ba760ddc3fca0828341f5d28157` — **EXACT MATCH** with the supplied production checkpoint file.
- `HANDOFF.md` (Line 15) records:
  `57b6a4be0eb1870bcae3eb999db495c6978434d2847fc7382d51b32637f8cf28`.
- **Explanation**: The hash in `HANDOFF.md` represents an earlier pre-calibration build before the validation temperature scaler and `has_validated_regression` metadata flags were injected directly into the checkpoint dictionary. The file supplied here is the **true, final evaluated champion checkpoint** (`57b6a4be8b14d11c2574c64192983942ae3d6ba760ddc3fca0828341f5d28157`) that matches `PERSON_2_FINAL_HANDOFF.md` and `evaluation/metrics/champion_test_report.json`.

---

## 3. Calibration & Regression Verification

### Embedded Checkpoint Payload Keys
The supplied `.pth` file is a complete Python dictionary saved with `torch.save`:
```python
checkpoint_keys = [
    'exp_name', 
    'model_state_dict', 
    'val_metrics', 
    'alpha', 
    'beta', 
    'history', 
    'temperature', 
    'has_validated_regression'
]
```

### Verification of Embedded Parameters:
- `checkpoint['temperature']`: `1.3391530513763428`
  - Computed via post-hoc Platt scaling / temperature scaling strictly on the held-out validation set (`data/splits/validation.csv`).
  - Exactly matches `champion_test_report.json` (`temperature_scaling_T: 1.3391530513763428`).
- `checkpoint['has_validated_regression']`: `True`
  - Signals to downstream consumers (`service_adapter.py`, `predictor.py`) that the continuous regression head is fully initialized and validated for conjunctiva.
- **Legacy `models/temperature_scaling.json` (3.063428)**:
  - The older JSON file containing `3.063428` was generated during preliminary single-head baseline experiments.
  - In our architecture, `src.inference.predictor.predict()` reads `temperature` directly from the `.pth` dictionary payload. The matching `temperature_scaling.json` with `1.3391530513763428` is included in this package for external consumers that parse JSON configs.

---

## 4. Verified Runtime Environment & Dependency Alignment

### Working Environment
- **Python Version**: `3.12.3` (compatible with 3.11+).
- **Core Working Dependencies** (`requirements-tested.txt`):
  - `torch`: `2.4.1+cu121` (or `torch>=2.4.0,<2.7.0` on CPU)
  - `torchvision`: `0.19.1+cu121` (or `torchvision>=0.19.0,<0.22.0`)
  - `numpy`: `2.4.6` (satisfies Person 1's `numpy>=2.2,<3`)
  - `scipy`: `1.15.3` (satisfies Person 1's `scipy>=1.15,<2`)
  - `Pillow`: `12.1.1` (satisfies Person 1's `Pillow>=11,<14`)
  - `opencv-python-headless`: `5.0.0.93` (satisfies Person 1's `opencv-python-headless>=4.11,<6`)
  - `Flask`: `3.1.3` (or `3.0.3`)
  - `Flask-SQLAlchemy`: `3.1.1`
  - `psycopg[binary]`: `3.3.6` (or `psycopg>=3.1,<4`)
  - `pandas`: `2.2.2`
  - `scikit-learn`: `1.7.1` (or `1.5.2`)
  - `reportlab`: `4.5.0`
  - `gunicorn`: `23.0.0`

### Conflict Resolution Summary
Older repository requirements had pinned `numpy==1.26.4`, `pillow==10.4.0`, and `opencv-python-headless==4.10.0.84`. Those clashed with `person1/pyproject.toml`. The versions above have been verified together across all 124 unit, pipeline, and UI integration tests with 0 failures.

---

## 5. Verification Commands & Sample Outputs

### Running Single-Image Inference
```bash
python -c "
from src.inference.predictor import predict
result = predict('samples/test_eye_conjunctiva.png', body_part='conjunctiva', metadata={'age': 24, 'sex': 'M'})
print(result)
"
```

### Verified Real Output 1: Eye Conjunctiva (`samples/test_eye_conjunctiva.png`)
```json
{
  "hb_estimate_g_dl": 10.77,
  "anemia_probability": 0.608,
  "predicted_class": "anemic",
  "model_version": "vyra-dualhead-efficientnet-b3-v1.0",
  "calibration_status": "calibrated",
  "quality_warnings": [],
  "error": null,
  "anatomical_modality": "conjunctiva",
  "ita_angle": -82.35,
  "ita_category": "Dark",
  "glare_fraction": 0.0002,
  "processing_time_ms": 694,
  "regression_status": "validated"
}
```
*Note: Ground truth reference venous Hb for this patient was 9.9 g/dL.*

### Verified Real Output 2: Fingernail (`samples/test_fingernail.png`)
```json
{
  "hb_estimate_g_dl": null,
  "anemia_probability": 0.567,
  "predicted_class": "anemic",
  "model_version": "vyra-dualhead-efficientnet-b3-v1.0",
  "calibration_status": "calibrated",
  "quality_warnings": [
    "Continuous Hb regression is validated strictly on conjunctiva with paired laboratory CBC; for fingernails, screening relies on the classification head."
  ],
  "error": null,
  "anatomical_modality": "fingernail",
  "ita_angle": -85.08,
  "ita_category": "Dark",
  "glare_fraction": 0.0,
  "processing_time_ms": 35,
  "regression_status": "unavailable"
}
```
*Note: Enforces strict clinical safety rule: continuous Hb is withheld (`null`, `"unavailable"`) for fingernail modality.*

---

## 6. Running Integration Verification Tests

Run the complete test suite against the production weights:
```bash
# Model architecture & unit tests
pytest tests/test_model_unit.py tests/test_vyra_pipeline.py

# Anatomical ROI detection & retake tests
pytest tests/test_roi_detector.py person3_app/tests/test_roi_detector.py

# Full Flask 3-panel UI & Service Adapter integration suite
pytest person3_app/tests
```
All test suites pass cleanly against these verified weights.
