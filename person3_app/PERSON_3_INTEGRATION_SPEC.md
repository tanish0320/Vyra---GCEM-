# Person 3 Integration Specification: VYRA AI/ML Inference Contract

This specification provides everything Person 3 needs to connect the Flask web application, patient intake workflow, and PDF report generator to the trained model.

---

## 1. Top-Level Inference Callable

Import the inference function from `src.inference.predictor`:

```python
from src.inference.predictor import predict

# Standard usage:
result = predict(
    image="path/to/image.png",       # str, Path, PIL.Image, or RGB numpy array
    feature_vector=None,             # Optional: 27-dim float vector from Person 1
    body_part="conjunctiva",         # "conjunctiva" or "fingernail"
    metadata={"age": 24, "sex": "F"},# Optional patient intake dictionary
    screening_threshold=0.50         # Default balanced threshold (use 0.255 for 100% recall triage mode)
)
```

---

## 2. Standardized Response Contract

The returned dictionary matches this exact schema:

```json
{
  "hb_estimate_g_dl": 6.48,
  "anemia_probability": 0.5497,
  "predicted_class": "anemic",
  "model_version": "vyra-dualhead-efficientnet-b3-v1.0",
  "calibration_status": "calibrated",
  "quality_warnings": [],
  "error": null,
  "anatomical_modality": "conjunctiva",
  "ita_angle": -78.23,
  "ita_category": "Dark",
  "glare_fraction": 0.0,
  "processing_time_ms": 35,
  "regression_status": "validated"
}
```

### Field Definitions & Units

| Field | Type | Units / Range | Description |
|---|---|---|---|
| `hb_estimate_g_dl` | `float` or `null` | g/dL (e.g. 6.48) | Continuous hemoglobin estimate. Only populated when clinically validated on paired CBC data (conjunctiva). Set to `null` for modalities without lab ground truth (fingernails). |
| `anemia_probability` | `float` or `null` | 0.0 to 1.0 | Post-hoc calibrated probability of anemia ($P(\text{Anemic})$). |
| `predicted_class` | `str` or `null` | `"anemic"`, `"non-anemic"` | Decision based on `screening_threshold`. |
| `model_version` | `str` | Text | Active model artifact version string. |
| `calibration_status` | `str` | `"calibrated"` or `"uncalibrated"` | Indicates whether Platt / temperature scaling was applied ($T = 1.5027$). |
| `quality_warnings` | `list[str]` | Text strings | Quality hints (e.g. excessive specular glare, low resolution, or unvalidated modality). |
| `error` | `str` or `null` | Text | Fatal error description if image failed to load or validate. |
| `anatomical_modality`| `str` | `"conjunctiva"`, `"fingernail"` | Modality parsed from user input. |
| `ita_angle` | `float` | Degrees ($-90^\circ$ to $+90^\circ$) | Individual Typology Angle computed in CIE L\*a\*b\* space. |
| `ita_category` | `str` | Category | Dermatological Fitzpatrick tier (`Very Light`, `Light`, `Intermediate`, `Tan`, `Brown`, `Dark`). |
| `glare_fraction` | `float` | 0.0 to 1.0 | Percentage of pixels identified as specular tear-film glare and inpainted. |
| `processing_time_ms`| `int` | Milliseconds | End-to-end execution latency. |
| `regression_status` | `str` | `"validated"`, `"unavailable"` | Status of the continuous Hb regression head. |

---

## 3. UI Display Guidelines for Person 3

1. **Hemoglobin Estimate Rule**:
   - Only display a numeric hemoglobin readout in the UI if `hb_estimate_g_dl is not None` and `regression_status == "validated"`.
   - If `hb_estimate_g_dl is None`, display: *"Continuous Hb estimate unavailable for this modality. Screening based on calibrated risk classification."*

2. **Triage Cards**:
   - **RED (Urgent Referral)**: If `hb_estimate_g_dl < 7.0` (or `anemia_probability > 0.85` in balanced mode).
   - **YELLOW (PHC Referral)**: If `7.0 <= hb_estimate_g_dl < 11.0` (or `anemia_probability >= 0.50`).
   - **GREEN (Routine Follow-up)**: If `hb_estimate_g_dl >= 11.0` and `anemia_probability < 0.50`.

3. **High-Sensitivity Triage Toggle**:
   - Pass `screening_threshold=0.255` when the CHW enables High-Sensitivity Mode (reaches 100% sensitivity for safety screening).

4. **Medical Disclaimer**:
   - *This tool is an AI-assisted screening prototype and does not constitute a confirmatory medical diagnosis. All flagged cases require laboratory Complete Blood Count (CBC) verification.*
