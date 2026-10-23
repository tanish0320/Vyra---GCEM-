# VYRA — Person 3, Phase 2: Integration & API Contracts (`PERSON3_PHASE2_API_CONTRACT.md`)

> **Status:** Proposed Integration Specification (Pending confirmation with Person 1 and Person 2).
> Nothing in this document claims that `normalization.py` or continuous Hb regression is already implemented in the repository.

---

## 1. Proposed Person 1 Preprocessing Interface (`normalization.py`)

### 1.1 Module Path & Callable Signature (Proposed)
- **Module Path:** `C:\Claude_projects\Vyra\anemia-detection\normalization.py`
- **Primary Entry Point (Proposal from Person 1 Workflow Plan):**
  ```python
  def normalize_image(
      image: str | Path | np.ndarray | PIL.Image.Image,
      body_part: str,
      *,
      mask: np.ndarray | str | Path | None = None,
      output_preview_path: str | Path | None = None,
  ) -> NormalizationResult | tuple[np.ndarray, list[float] | None, dict[str, Any]]:
  ```
  *(Exact return container—dataclass vs. 3-tuple `(normalized_image, feature_vector, metadata)`—to be confirmed with Person 1 and Person 2.)*

### 1.2 Input Contract
| Parameter | Proposed Type | Proposed Semantics & Constraints |
| :--- | :--- | :--- |
| `image` | `Path \| str \| np.ndarray \| PIL.Image.Image` | Raw uploaded image **before** `app.py`'s square `ImageOps.fit` resize so surrounding epidermis and aspect ratio are preserved. Expected color space: **RGB** (`uint8`, range `[0, 255]`, 3 channels `[H, W, 3]`) if passed as array/PIL, or file path on disk. (If OpenCV `BGR` is preferred internally by Person 1, the boundary conversion must be explicit.) |
| `body_part` | `str` | Canonical anatomical modality token. Proposed canonical values mapped from UI `IMAGE_TYPE_OPTIONS`: `"conjunctiva"` (from `"Eye Conjunctiva"`) and `"fingernail"` (from `"Fingernail"`). Unsupported values (including `"Unknown"`) must raise a structured `ValueError` or return `is_valid=False` with a clear error message. |
| `mask` | Optional `Path \| np.ndarray \| None` | Optional ROI segmentation mask when evaluating dataset images that include ground-truth masks; `None` during live smartphone capture/upload (triggering Person 1's threshold/crop or alignment fallback). |
| `output_preview_path` | Optional `Path \| None` | Optional destination under `static/uploads/` (or `static/normalized/`) where Person 3's adapter can persist the normalized ROI preview image for the UI's 3-stage visual inspection panel (`Raw -> Normalized ROI -> Grad-CAM`). |

### 1.3 Order of Operations (Per Source Workflow Plan)
1. **ROI Isolation / Cropping & Glare Masking:** Isolate palpebral conjunctiva or fingernail bed ROI; detect specular highlight/glare pixels and mask or inpaint where feasible.
2. **Grey-World White Balance:** Compensate for ambient illuminant color cast.
3. **HSV Value-Channel CLAHE:** Convert to HSV, apply CLAHE on the `V` channel only, and convert back to RGB so chromaticity (`H`, `S`) is not distorted.
4. **CIE LAB ITA Calculation & Colorimetric Feature Extraction:**
   - Compute Individual Typology Angle ($\text{ITA}^\circ = \arctan\left(\frac{L^* - 50}{b^*}\right) \times \frac{180}{\pi}$) from surrounding epidermis pixels when present.
   - Extract colorimetric features referenced in Person 1's plan: `R/G` ratio, `R/B` ratio, High-Hue Ratio (`HHR`), and pallor/spatial gradient features.

### 1.4 Output Contract (Proposed)
1. **`normalized_image`:**
   - **Type & Shape:** `np.ndarray` (`uint8`, RGB `[H, W, 3]`) or `PIL.Image.Image` (`mode="RGB"`), ready for Person 2's `eval_transform` (which resizes/crops to `(300, 300)` or `(224, 224)` and normalizes with `IMAGENET_MEAN`/`IMAGENET_STD`) and for saving as the UI preview image.
2. **`feature_vector`:**
   - **Type:** `list[float]` or 1D `np.ndarray` (`dtype=np.float32`) in a fixed, documented order, OR `None` if image rejection occurred.
   - **Feature Names, Length & Units:** *Pending confirmation between Person 1 and Person 2.* Do not hardcode vector length in `app.py`; pass `feature_vector` and `feature_names` through the adapter to Person 2's predictor.
3. **`metadata` (`dict[str, Any]`):**
   ```json
   {
     "is_valid": true,
     "body_part": "conjunctiva",
     "roi_method": "auto_crop_fallback",
     "glare_pixel_ratio": 0.018,
     "glare_inpainted": true,
     "ita_degrees": null,
     "ita_category": null,
     "ita_available": false,
     "ita_unavailable_reason": "Pre-cropped ROI image; surrounding epidermis not present.",
     "feature_names": ["r_g_ratio", "r_b_ratio", "hhr"],
     "feature_dict": {},
     "normalized_preview_path": "uploads/20261023_120000_norm_ab12cd34.jpg",
     "warnings": [],
     "error": null
   }
   ```

### 1.5 Missing-Data & Failure Rules
- **Missing Surrounding Epidermis / Pre-Cropped ROI:** Must set `ita_degrees: null`, `ita_available: false`, and populate `ita_unavailable_reason` rather than inventing an ITA value.
- **Missing Segmentation Mask:** Must fall back to deterministic ROI crop/thresholding and record `roi_method` in `metadata`.
- **Unusable / Invalid Image (corrupt file, grayscale/1-channel, `<100x100` px, extreme glare obscuring ROI, or unsupported `body_part`):** Must not crash Flask; must return `is_valid=False` and a human-readable `error` string (or raise a typed `NormalizationError` caught by Person 3's service adapter).

---

## 2. Proposed Person 2 Inference Interface (`model.py` & `predict.py`)

### 2.1 Current Baseline vs. Target Person 2 Architecture
- **Current Baseline in Repo (`model.py` & `predict.py`):**
  - `build_model()` constructs `efficientnet_b3` (or `efficientnet_b0` on CPU fast profile) with a single `nn.Sequential` classifier head (`1536 -> 512 -> 256 -> 2`) outputting 2 binary logits (`CLASS_NAMES = ("non-anemic", "anemic")`).
  - **No continuous Hb regression head exists in `model.py`.**
  - `predict.py:estimate_hemoglobin_range()` returns a static string band (`"7-9 g/dL (estimated)"`, etc.) derived purely from the binary prediction and confidence. **Per project rules, this heuristic string must not be presented as a continuous regression Hb estimate.**
- **Target Person 2 Architecture (Per Person 2 Workflow Plan):**
  - Dual-head architecture: binary classification (`anemic` vs. `non-anemic`) + positive continuous hemoglobin regression head (`hb_estimate_g_dl` in `g/dL`, constrained positive via `Softplus` or equivalent), optionally fusing the EfficientNet visual embedding with Person 1's colorimetric `feature_vector` and `body_part` encoding.
  - Fallback rule (from Person 2 plan): If labelled Hb data/time cannot support validated regression within 24 hours, preserve the working binary classifier and return `hb_estimate_g_dl: null` (`hb_regression_available: false`) rather than fabricating numeric Hb values or deriving numeric Hb from a class label.

### 2.2 Proposed Inference Callable Signature
```python
def predict_from_normalized(
    normalized_image: Path | np.ndarray | PIL.Image.Image,
    *,
    raw_image_path: Path | None = None,
    feature_vector: Sequence[float] | np.ndarray | None = None,
    body_part: str = "conjunctiva",
    patient_info: dict[str, Any] | None = None,
    save_gradcam: bool = True,
) -> dict[str, Any]:
```
*(Should also preserve backward compatibility with `AnemiaPredictor.predict_image(image_path, patient_info=..., save_gradcam=...)`.)*

### 2.3 Proposed Person 2 Output Schema
```json
{
  "predicted_class": "Anemic",
  "anemia_probability": 0.842,
  "non_anemic_probability": 0.158,
  "confidence": 0.842,
  "hb_estimate_g_dl": null,
  "hb_regression_available": false,
  "hb_units": "g/dL",
  "hb_interval_g_dl": null,
  "model_version": "efficientnet_b3_v1",
  "architecture": "efficientnet_b3",
  "calibration_status": "temperature_scaled",
  "calibration_temperature": 3.0634,
  "gradcam_path": "C:/Claude_projects/Vyra/anemia-detection/static/gradcam/scan_gradcam_1a2b3c4d.png",
  "gradcam_used_fallback": false,
  "processing_time_ms": 410,
  "warnings": [
    "Continuous hemoglobin regression head is not enabled in this checkpoint; showing classification probability only."
  ],
  "error": null
}
```

### 2.4 Field Semantics & Non-Fabrication Guarantees
| Field | Type | Rules & Constraints |
| :--- | :--- | :--- |
| `hb_estimate_g_dl` | `float \| null` | Positive float in `g/dL` **only** when produced by a trained regression head. **Must be `null`** when using a classification-only checkpoint. Never parse or midpoint-convert the baseline's `"7-9 g/dL (estimated)"` string into a float. |
| `anemia_probability` | `float` | Probability of `"anemic"` class in `[0.0, 1.0]` (maps to existing `anemic_probability`). |
| `predicted_class` | `str \| null` | `"Anemic"` or `"Non-Anemic"` (or `null` if inference failed). |
| `model_version` | `str` | Identifier of checkpoint filename + epoch/hash (e.g. `"20260428_2028_best_model:epoch3"`). |
| `calibration_status` | `str` | One of `"temperature_scaled"`, `"uncalibrated"`, `"validation_calibrated"`, or `"unavailable"`. Never claim validation calibration unless fitted on a leakage-safe validation split. |
| `gradcam_path` | `str \| null` | Local file path or storage reference (`gradcam/<filename>`) to the generated Grad-CAM image, or `null` if disabled/unavailable. |
| `warnings` | `list[str]` | List of non-fatal model/calibration/Grad-CAM warnings. |
| `error` | `str \| null` | Non-null error description if inference failed. |

### 2.5 Evaluation Artifacts Contract (For `/analytics` and `/about`)
- **Locations checked by `app.py:performance_overview()`:**
  - `logs/evaluation/test_summary.json` (generated by `evaluate.py --split test`)
  - `logs/final_test_summary.json` (generated by `train.py`)
- **Required Additional Evaluation Artifacts from Person 2 (When Supported by Data):**
  - Regression metrics (`mae_g_dl`, `rmse_g_dl`), Bland-Altman plot (`bland_altman.png`), reliability/calibration curve (`calibration_curve.png`), confusion matrix (`confusion_matrix.png`), and dataset split provenance (`dataset_provenance` with patient-level split counts).
  - If an artifact is not generated, `/about` and `/analytics` must display `"Unavailable / Not evaluated"` rather than falling back to `DEFAULT_CLINICAL_ACCURACY = 92.4` or `99.7%`.

---

## 3. Proposed Person 3 Flask Orchestration & Service Adapter Contract

### 3.1 Service Adapter Design (`service_adapter.py` — Planned for Phase 3)
Flask routes (`POST /predict` and `POST /api/predict`) in `app.py` will not call `model.py`, `normalization.py`, or `predict.py` directly. Instead, they will call a single orchestration service (`run_screening_pipeline(...)`) with the following sequence:

1. **Save Raw Upload Without Destructive Cropping:**
   - Save the original EXIF-transposed RGB image to `static/uploads/` preserving its native resolution (up to a safe maximum dimension, e.g. `1600px`, without upscaling `<100x100` images or square-cropping before validation/normalization).
2. **Run `image_validator.validate_image(raw_image_path)`:**
   - If `not validation.is_valid`, immediately return HTTP `400` (`{"success": False, "error": validation.error, "quality": quality_payload}`).
3. **Check Model / Pipeline Readiness (LFS & Load-Error Aware):**
   - Inspect checkpoint file header before `torch.load()`: if the file begins with `version https://git-lfs.github.com/spec/v1` or `_predictor_load_error` is set, return HTTP `503` with a truthful diagnostic message:
     - `"Model checkpoint unavailable or invalid (Git LFS pointer detected). Real inference cannot run until a valid .pth checkpoint is installed."`
   - Only if `VYRA_ALLOW_DEV_MOCK=true` is explicitly set in development, allow a clearly labelled mock response with `"is_mock": true` and `"model_version": "DEV_MOCK_NOT_FOR_CLINICAL_USE"`. Default in production/demo is `VYRA_ALLOW_DEV_MOCK=false`.
4. **Invoke Person 1 Preprocessing (`normalize_image`):**
   - Pass `raw_image_path` and mapped `body_part` (`"conjunctiva"` or `"fingernail"`).
   - Persist the normalized ROI preview image so the result screen can show `Raw Image -> Normalized ROI + ITA Badge -> Grad-CAM`.
   - If `normalization.py` is not yet merged by Person 1, log a warning and pass the validated raw image to Person 2's predictor with `normalization_applied: false` in the response.
5. **Invoke Person 2 Inference:**
   - Pass `normalized_image`, `feature_vector`, `body_part`, and `patient_info`.
6. **Compute CHW Triage & Referral Recommendation:**
   - Map model outputs + demographic inputs (`age`, `gender`, `pregnancy_status` if provided) into `Green` / `Yellow` / `Red` triage state once threshold rules are confirmed by the team, always accompanied by the confirmatory CBC test disclaimer.
7. **Persist Scan & Generate PDF:**
   - Save record to SQLite (`scans` + `media_files`), and expose updated referral PDF via `/export/pdf/<scan_id>`.

### 3.2 Backward-Compatible `POST /api/predict` Response Schema
To avoid breaking existing frontend JS or API callers, all 26 existing keys in `build_prediction_response()` are retained, and new VYRA integration fields are added alongside them:

```json
{
  "success": true,
  "is_mock": false,
  "id": 42,
  "scan_id": 42,
  "public_scan_id": "AV-000042",
  "persisted": true,
  "patient_name": "Aarav Kumar",
  "prediction": "Anemic",
  "predicted_class": "Anemic",
  "confidence": 0.842,
  "confidence_percent": 84.2,
  "anemic_probability": 0.842,
  "anemia_probability": 0.842,
  "non_anemic_probability": 0.158,
  "hb_estimate_g_dl": null,
  "hb_regression_available": false,
  "hemoglobin_estimate": "Unavailable (binary classifier checkpoint)",
  "model_version": "efficientnet_b3_v1",
  "calibration_status": "temperature_scaled",
  "risk_level": "Medium Risk",
  "risk_explanation": "The scan shows meaningful anemia-related cues. A doctor visit and confirmatory lab testing are recommended.",
  "triage": {
    "color": "Yellow",
    "label": "Moderate Screening Alert — Refer for Confirmatory CBC",
    "referral_facility": "Primary Health Centre (PHC)",
    "referral_status": "Pending Referral",
    "prototype_disclaimer": "Screening aid prototype only. Confirm with laboratory CBC/hemoglobin testing."
  },
  "normalization": {
    "applied": true,
    "body_part": "conjunctiva",
    "roi_method": "auto_crop",
    "ita_degrees": null,
    "ita_category": null,
    "ita_available": false,
    "normalized_image_url": "/media/uploads/20261023_120000_norm_ab12cd34.jpg"
  },
  "medical_advice": "Please consult a doctor or clinic soon for a full medical assessment...",
  "quality": {
    "label": "Good quality",
    "icon": "check",
    "is_good": true,
    "score": 118.49,
    "brightness": 114.34,
    "warning": null,
    "warnings": [],
    "error": null,
    "image_quality": "Good"
  },
  "image_quality": "Good",
  "quality_warning": null,
  "quality_warnings": [],
  "image_url": "/media/uploads/20261023_120000_raw_ab12cd34.jpg",
  "gradcam_url": "/media/gradcam/20261023_120000_gradcam_1a2b3c4d.png",
  "result_url": "/result/42",
  "pdf_url": "/export/pdf/42",
  "processing_time_ms": 485,
  "warnings": [],
  "error": null
}
```

### 3.3 HTTP Status Codes & Error Contract
| Scenario | HTTP Status | JSON Payload (`POST /api/predict`) | HTML Form Behavior (`POST /predict`) |
| :--- | :--- | :--- | :--- |
| Valid request & prediction succeeded | `200 OK` | Full response schema above (`"success": true`) | Renders `result.html` with flash `"Scan completed successfully."` |
| Missing/invalid patient fields (`patient_name`, `age`, `gender`) | `400 Bad Request` | `{"success": false, "errors": ["Age is required.", ...]}` | Flashes each validation error (`warning`) and redirects (`302`) to `/scan` |
| Missing image, unsupported format, `<100x100` px, or rejected by Person 1 validator | `400 Bad Request` | `{"success": false, "error": "<exact validation reason>", "quality": {...}}` | Flashes error (`danger`) and redirects (`302`) to `/scan` |
| Rate limit exceeded (`>10` req/min/IP) | `429 Too Many Requests` | `{"success": false, "error": "Too many prediction requests..."}` + `Retry-After` header | Flashes warning and redirects (`302`) to `/scan` |
| Model checkpoint missing or Git LFS pointer / corrupt checkpoint | `503 Service Unavailable` | `{"success": false, "error": "Model checkpoint failed to load: <reason>", "checkpoint_valid": false}` | Flashes clear error (`danger`) and redirects (`302`) to `/scan` |
| Model warmup genuinely in progress | `503 Service Unavailable` | `{"success": false, "error": "The AI model is still starting on this server..."}` | Flashes warning and redirects (`302`) to `/scan` |
| Unhandled internal exception | `500 Internal Server Error` | `{"success": false, "error": "An internal server error occurred. Please try again shortly."}` | Rolls back DB session and renders `500.html` |

---

## 4. Unresolved Questions Requiring Team Agreement

1. **Person 1 + Person 2:** What is the exact ordered list of colorimetric features in `feature_vector` (e.g., `[r_g_ratio, r_b_ratio, hhr, l_mean, a_mean, b_mean, ita_degrees, ...]`), its length, and how should missing `ita_degrees` be encoded inside `feature_vector` (e.g., `0.0` + binary indicator flag `ita_present`, or excluded from the tensor)?
2. **Person 1 + Person 3:** Will `normalize_image(image, body_part)` accept a `Path` or `np.ndarray` (RGB vs. BGR), and will Person 1 save the normalized ROI preview image or return an `np.ndarray` (`uint8` RGB) for Person 3's `storage_utils` to save?
3. **Person 2 + Person 3:** Will the Phase 3 checkpoint include a trained continuous Hb regression head (`hb_estimate_g_dl: float`), or should Phase 3 initially integrate against the binary classifier checkpoint (`hb_estimate_g_dl: null`) while regression training completes?
4. **All Three Teammates (Clinical Triage Rules):** When `hb_estimate_g_dl` is available vs. unavailable (classification probability only), what exact thresholds define **Green** (Low Risk / Routine nutrition counseling), **Yellow** (Moderate Risk / Non-urgent PHC referral for CBC), and **Red** (High Risk / Urgent FRU/District Hospital referral), and should pregnancy status (`pregnant: bool`) be added to the `/scan` intake form to support WHO demographic Hb cutoffs (`11.0 g/dL` pregnant/children <5y, `12.0 g/dL` non-pregnant women, `13.0 g/dL` men)?
