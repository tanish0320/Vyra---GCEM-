# VYRA — Person 3, Phase 2: Teammate Handoff & File Ownership (`PERSON3_PHASE2_HANDOFF.md`)

---

## 1. Proposed File Ownership Matrix & Conflict Mitigation

> **Note:** This ownership assignment is a **proposal** based on the three workflow plans and has not yet been formally approved by Person 1 and Person 2.

| File / Directory | Primary Owner | Secondary / Consumer | Conflict Risk | Integration & Edit Rules |
| :--- | :--- | :--- | :--- | :--- |
| `normalization.py` *(planned new module)* | **Person 1** | Person 2 (training features), Person 3 (live adapter) | **Low** | Person 1 owns exclusively. Exposes `normalize_image(image, body_part)`. |
| `data_manifest.csv`, split manifests, leakage audit scripts *(planned)* | **Person 1** | Person 2 (`dataset.py`), Person 3 (`/about` metrics) | **Low** | Person 1 owns exclusively. |
| `image_validator.py` | **Person 1 & Person 3 (Shared)** | `app.py`, `predict.py` | **Medium** | Existing functions `validate_image()` and `build_quality_payload()` must remain backward-compatible. Any new glare/ROI quality gates should either live in `normalization.py` or extend `ImageValidationResult` with default-valued fields. |
| `dataset.py`, `train.py`, `evaluate.py` | **Person 2** | Person 1 (split manifests) | **Low** | Person 2 owns exclusively. Person 3 does not modify training/evaluation scripts. |
| `model.py` | **Person 2** | Person 3 (via adapter / `predict.py`) | **Medium** | Person 2 owns model architecture (`build_model`, dual-head classification + Hb regression, `load_checkpoint`). Must preserve or clearly version `load_checkpoint()`. |
| `models/*.pth`, `models/temperature_scaling.json`, `logs/evaluation/*` | **Person 2** | Person 3 (`/health`, `/about`, `/analytics`) | **Medium** | Person 2 supplies valid `.pth` weights and real evaluation JSON/PNG artifacts. |
| `predict.py`, `gradcam.py` | **Person 2 & Person 3 (Shared)** | `app.py`, `pdf_report.py` | **HIGH** | **Highest conflict zone.** `app.py` and `pdf_report.py` import `AnemiaPredictor`, `PredictionResult`, `compute_risk_metadata`, `estimate_hemoglobin_range`, and `medical_advice_lines` from `predict.py`. Person 2 should update model forward/calibration logic inside `predict.py`, while Person 3 wraps orchestration in a new `service_adapter.py` rather than rewriting `predict.py` simultaneously. |
| `config.py` | **Shared (All 3)** | All modules | **HIGH** | Additive constants only. Do not rename existing exports (`BEST_CHECKPOINT_PATH`, `UPLOADS_DIR`, `GRADCAM_DIR`, `REPORTS_DIR`, `FLASK_CONFIG`, `TRAINING_CONFIG`). |
| `service_adapter.py` *(planned new module)* | **Person 3** | Calls Person 1 & Person 2 | **Low** | Person 3 owns exclusively; isolates `app.py` route handlers from preprocessing and model internals. |
| `app.py`, `schema_patches.py`, `migrate.py`, `storage_utils.py`, `pdf_report.py` | **Person 3** | None | **Low** | Person 3 owns Flask routes, DB persistence, FTS5 search, CHW triage logic, and ReportLab PDF generation. |
| `templates/*`, `static/css/*`, `static/js/*` | **Person 3** | None | **Low** | Person 3 owns UI templates, mobile capture workflow, result presentation, and analytics views. |

---

## 2. Git LFS Checkpoint Blocker — Exact Technical Analysis

### 2.1 Inspected Checkpoint Files on Disk
Both checkpoint files tracked in `.gitattributes` are **134-byte Git LFS pointer text files**, not binary PyTorch archives, because `RAHULPATEL2002/anemia-detection` exceeded its GitHub LFS bandwidth quota (`batch response: This repository exceeded its LFS budget`):

1. **`C:\Claude_projects\Vyra\anemia-detection\models\20260428_2028_best_model.pth`**
   - **Actual Size on Disk:** `134` bytes (expected binary size: `140,300,969` bytes / ~140.3 MB)
   - **Exact File Contents:**
     ```text
     version https://git-lfs.github.com/spec/v1
     oid sha256:3b3e4d2e32f2e3cf6529e96cda6938f79619439282c84a108a66126eb0110f70
     size 140300969
     ```
2. **`C:\Claude_projects\Vyra\anemia-detection\models\20260428_2028_last_checkpoint.pth`**
   - **Actual Size on Disk:** `134` bytes (expected binary size: `140,315,495` bytes / ~140.3 MB)
   - **Exact File Contents:**
     ```text
     version https://git-lfs.github.com/spec/v1
     oid sha256:1865d9c8e8cc2e21028d9b2ed66d01b1b55cb854239cef15f30a7786299895f0
     size 140315495
     ```
3. **`C:\Claude_projects\Vyra\anemia-detection\models\temperature_scaling.json`**
   - **Actual Size on Disk:** `41` bytes (valid JSON, non-LFS): `{"temperature": 3.0634281635284424}`

### 2.2 Failure Chain in Existing Code
1. **`model.py:latest_available_checkpoint()` (`lines 182–188`):**
   - Checks `if checkpoint_path.exists(): return checkpoint_path`. Because the 134-byte LFS pointer file exists on disk, `latest_available_checkpoint()` returns `Path(".../models/20260428_2028_best_model.pth")`.
2. **`app.py:eager_load_predictor()` (`lines 311–321`) & `get_predictor()` (`lines 277–302`):**
   - Calls `AnemiaPredictor(checkpoint_path=checkpoint)` -> `load_checkpoint()` -> `torch.load(checkpoint_path, map_location=device, weights_only=False)`.
   - `torch.load()` attempts to unpickle the file starting with ASCII `'v'` (`version https://git-lfs...`) and raises:
     `_pickle.UnpicklingError: invalid load key, 'v'.`
   - `get_predictor()` stores `_predictor_load_error = "invalid load key, 'v'."` and leaves `_predictor = None`.
3. **`GET /health` (`app.py:1380–1402`):**
   - Reports `"checkpoint_available": true`, `"predictor_loaded": false`, and `"predictor_load_error": "invalid load key, 'v'."`.
   - Client JS (`static/js/main.js:88` and `static/js/upload.js:306–315`) checks `if (payload.predictor_loaded) ... else if (payload.checkpoint_available)` and displays **"Model Waking" / "Model Loading"** forever while keeping the `#submitScanButton` disabled!
4. **`POST /api/predict` (`app.py:1508–1525`):**
   - Checks `if not predictor_loaded(): maybe_start_predictor_warmup(); return jsonify({"success": False, "error": "The AI model is still starting on this server. Please wait a few seconds and try again."}), 503`.

### 2.3 Additional Finding From `TRAINING_GUIDE.md` & `logs/training_history.json`
- According to `TRAINING_GUIDE.md` (`lines 3–15`) and `logs/training_history.json`, the upstream checkpoint was trained for only **3 epochs on CPU** (`fast` profile, `efficientnet_b0`, `224x224`), reaching `valid_accuracy = 0.4486` (`44.9%`) and `valid_auc_roc = 0.5826`.

### 2.4 Recommended Recovery Options
1. **Option 1 (Preferred for Real Inference):** Person 2 trains or supplies a valid PyTorch `.pth` checkpoint (or places `models/best_model.pth` locally, which is ignored by Git LFS if named `best_model.pth` or if `.gitattributes` is updated on the integration branch).
2. **Option 2 (Baseline Classifier Fallback):** Person 2 initializes a valid checkpoint file via `model.build_model(pretrained=True)` + `model.save_checkpoint()` so the pipeline can execute real forward passes and Grad-CAM generation while fine-tuning is underway (explicitly marked as `uncalibrated / baseline_imagenet_init` in `calibration_status`).
3. **Option 3 (Checkpoint Integrity Validation in Phase 3):** Update checkpoint verification so any file `< 1024` bytes or starting with `b"version https://git-lfs.github.com/spec/v1"` is flagged as `checkpoint_valid = False` with a clear error message in `/health` and `/api/predict`.
4. **Option 4 (Explicit Development Mock Mode for UI Testing in Phase 3):** In Person 3's `service_adapter.py`, support an opt-in environment variable (`VYRA_ALLOW_DEV_MOCK=true`) that returns a deterministic, explicitly badged `[DEV MOCK — NOT A MODEL PREDICTION]` response so frontend/triage/PDF workflows can be tested before Person 2's checkpoint is ready.

---

## 3. Handoff Questions for Teammates

### 3.1 Questions for Person 1 (Data, Computer Vision & Optical Normalization)
1. **Function Signature & Return Type:** Do you confirm `normalize_image(image, body_part)` in `normalization.py` returning `(normalized_image, feature_vector, metadata)`, and what types will `image` and `normalized_image` use (`np.ndarray` RGB `uint8` vs. `PIL.Image.Image` vs. `Path`)?
2. **Canonical `body_part` Values:** Currently the Flask UI sends `"Eye Conjunctiva"`, `"Fingernail"`, or `"Unknown"`. Should Person 3's adapter map these to `"conjunctiva"` and `"fingernail"` before calling `normalize_image()`, and should `"Unknown"` be rejected or default to `"conjunctiva"`?
3. **Raw Image Resolution Requirement:** `app.py:optimize_and_save_image()` currently center-crops and resizes uploads to `224x224` *before* validation/inference. Should Person 3 disable pre-crop resizing on the raw upload so `normalize_image()` receives the full uncropped smartphone photo (including surrounding epidermis for ITA calculation)?
4. **Feature Vector Specification:** What are the exact feature names, length, order, and numeric ranges in `feature_vector`, and how is missing ITA (when surrounding skin is absent in pre-cropped dataset images) represented?
5. **Normalized Preview Persistence:** Should `normalize_image()` accept an `output_preview_path` to save the normalized ROI image directly, or return the RGB `np.ndarray` for Person 3 to save under `static/uploads/`?

### 3.2 Questions for Person 2 (AI/ML, Hemoglobin Estimation & Evaluation)
1. **Checkpoint Delivery & Format:** When will a valid non-LFS `.pth` checkpoint be available, what filename will it use (`models/best_model.pth`?), and what keys will its checkpoint dictionary contain (`model_state_dict`, `architecture`, `epoch`, `metrics`, `config`, `temperature`)?
2. **Dual-Head vs. Classifier Fallback:** Will the checkpoint include the continuous Hb regression head (`hb_estimate_g_dl` in `g/dL` with `Softplus`), or should Phase 3 initially expect binary classification outputs only (`hb_estimate_g_dl = None`)?
3. **Feature Fusion Input:** Does your inference callable require `(normalized_image, feature_vector, body_part)` from Person 1, or does it run on `normalized_image` alone if `feature_vector` is `None`?
4. **Input Resolution & Architecture:** Will inference use `efficientnet_b3` at `(300, 300)` or `efficientnet_b0` at `(224, 224)`? (Note that `config.py` currently switches `MODEL_ARCH` and `image_size` automatically based on whether a GPU is present unless overridden in the checkpoint metadata.)
5. **Evaluation Artifacts for UI:** Where will you write the final held-out test metrics and plots (`logs/evaluation/test_summary.json`, `confusion_matrix.png`, `roc_curve.png`, `calibration_curve.png`, `bland_altman.png`), and what keys will `test_summary.json` contain for MAE, RMSE, sensitivity, specificity, and split provenance?

---

## 4. Confirmed Decisions vs. Pending Decisions

### Confirmed Decisions (Verified in Phase 1 & Phase 2)
- **Base Scaffold:** Retain `RAHULPATEL2002/anemia-detection` Flask/Jinja2/Vanilla JS application in `C:\Claude_projects\Vyra\anemia-detection`.
- **Runtime Environment:** Python 3.11 `.venv` using dependencies from `requirements-full.txt`.
- **Database & PDF Engine:** Retain SQLite (`database/anemia_vision.db` with WAL + FTS5 `scan_name_fts`) and ReportLab (`pdf_report.py`), extending them additively via `schema_patches.py`.
- **Strict Non-Fabrication Policy:** Do not present `predict.py:estimate_hemoglobin_range()` string bands as regression Hb outputs; do not display hardcoded `92.4%` / `99.7%` accuracy or `10,000+` scan counts as team evaluation results; label any development mock unmistakably.

### Pending Decisions (Require Teammate Confirmation)
- Exact `normalize_image(image, body_part)` return signature and `feature_vector` ordering (Person 1 & Person 2).
- Availability of continuous Hb regression head vs. binary classifier fallback in the checkpoint (Person 2).
- Exact demographic-aware Green / Yellow / Red triage thresholds (whether based on continuous Hb cutoffs, calibrated `anemia_probability`, or a combination of both) and whether `pregnancy_status` should be added to the `/scan` intake form (All 3 teammates).

---

## 5. Phase 3 Readiness & Prerequisites

### Ready Now for Person 3 Phase 3 (H3–H8: UI & Mock-Backed Workflow)
- Complete route, template, static JS, SQLite schema, and PDF report inventory is documented.
- Exact bugs in baseline `app.py` (pre-validation image cropping in `optimize_and_save_image()`, and `/health` + `/api/predict` ignoring `_predictor_load_error` on LFS pointer files) are pinpointed with line numbers.
- Backward-compatible API and service-adapter contracts are specified.

### Blocking Dependencies Before Real End-to-End Inference (H6–H15)
1. **Valid Model Checkpoint from Person 2** (to replace the 134-byte Git LFS pointers in `models/`).
2. **`normalization.py` Module from Person 1** (to provide real ROI crop, glare masking, Grey-World white balance, HSV CLAHE, and ITA metadata).
   - *Mitigation for Phase 3:* Person 3 can build the `service_adapter.py`, intake/result UI updates, and clearly-labelled development mock fallback immediately in Phase 3 without being blocked by Person 1 or Person 2.
