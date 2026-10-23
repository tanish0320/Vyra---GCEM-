# VYRA — Person 3, Phase 1: Baseline Setup & Verification Report

## 1. Workspace and Repository Location

- **Workspace Root Directory:** `C:\Claude_projects\Vyra`
  - **Git Remote (`origin`):** `https://github.com/tanish0320/Vyra---GCEM-.git`
  - **Active Branch:** `Tanish` (with `feature/vyra-integration` also created at `HEAD`)
  - **Project Workflow Documents Present in Workspace Root:**
    1. `Person_1_Data_Computer_Vision_Normalization_Plan.docx` (source-aligned to *Anemia Screening Hackathon Project Guide*)
    2. `Person_2_AI_Model_Hb_Estimation_Evaluation_Plan.docx` (source-aligned to *Anemia Screening Hackathon Project Guide*)
    3. `Person_3_Full_Stack_Integration_CHW_Workflow_Plan.docx` (source-aligned to *Anemia Screening Hackathon Project Guide*)
    - *Note:* The standalone *Anemia Screening Hackathon Project Guide* file is not stored separately in `C:\Claude_projects\Vyra`; its requirements and guardrails are embedded in the three source-aligned workflow documents.
- **Cloned Upstream Scaffold Repository:** `C:\Claude_projects\Vyra\anemia-detection`
  - **Upstream Remote (`origin`):** `https://github.com/RAHULPATEL2002/anemia-detection.git`
  - **Active Branch:** `feature/vyra-integration` (with `Tanish` also created at `HEAD`, commit `bb87141` / `origin/main`)

---

## 2. Environment and Dependency Setup Instructions

### Toolchain Verified
- **Git:** `git version 2.55.0.windows.2` (`C:\Program Files\Git\cmd\git.exe`)
- **Python Launcher (`py`):** Python 3.11 (64-bit), Python 3.13 (64-bit), and Python 3.14.2 installed; `.python-version` in the upstream repo specifies `3.12`. Python `3.11` (64-bit) was selected for `.venv` to ensure full wheel compatibility across PyTorch, OpenCV, ReportLab, and Flask-SQLAlchemy.

### Dependency Manifest Findings
- `requirements.txt` (14 bytes) only pins `flask==3.0.3` for the lightweight Vercel serverless UI entrypoint (`api/index.py`).
- `requirements-full.txt` (304 bytes, used by `Dockerfile`) contains the actual full application stack:
  - `torch==2.5.1`, `torchvision==0.20.1`
  - `flask==3.0.3`, `flask-sqlalchemy==3.1.1`
  - `psycopg[binary]>=3.1,<4`
  - `opencv-python-headless==4.10.0.84`, `pillow==10.4.0`
  - `numpy==1.26.4`, `pandas==2.2.3`, `scikit-learn==1.5.2`
  - `matplotlib==3.9.2`, `seaborn==0.13.2`
  - `reportlab==4.2.5`, `tqdm==4.66.6`, `colorama==0.4.6`, `gunicorn==23.0.0`

### Reproducible Setup Commands (Windows PowerShell)
```powershell
# 1. Clone scaffold with Git LFS smudge skipped (due to upstream GitHub LFS quota exhaustion)
$env:GIT_LFS_SKIP_SMUDGE="1"
git clone https://github.com/RAHULPATEL2002/anemia-detection.git C:\Claude_projects\Vyra\anemia-detection
Set-Location C:\Claude_projects\Vyra\anemia-detection

# 2. Create project-local virtual environment (Python 3.11)
py -3.11 -m venv .venv --system-site-packages

# 3. Install required packages from requirements-full.txt
.\.venv\Scripts\python.exe -m pip install flask-sqlalchemy==3.1.1 reportlab==4.2.5 "psycopg[binary]>=3.1,<4" gunicorn==23.0.0
```

---

## 3. Application Startup Command & Local URL

### Startup Command
```powershell
Set-Location C:\Claude_projects\Vyra\anemia-detection
.\.venv\Scripts\python.exe app.py
```

### Verified Local URL
- **Base URL:** `http://127.0.0.1:5000`
- **Health Endpoint:** `http://127.0.0.1:5000/health`

---

## 4. Existing Routes and Component Inventory

### Core Python Modules
| File | Exists? | Actual Responsibility |
| :--- | :--- | :--- |
| `app.py` | Yes (60,352 B) | Main Flask application (`create_app()`), SQLAlchemy models (`Scan`, `MediaFile`, `RateLimitEvent`), SQLite WAL & FTS5 search index (`scan_name_fts`), per-IP rate limiting, HTML & JSON routes. |
| `config.py` | Yes (16,892 B) | Central paths, device detection (`cuda` -> `mps` -> `cpu`), `TrainingConfig`, `AugmentationConfig`, `FlaskConfig`, SQLite/Postgres URI builder, runtime directory creation. |
| `image_validator.py` | Yes (6,600 B) | Pre-inference quality gate (`validate_image`, `build_quality_payload`). Rejects unsupported extensions (allowed: `.jpg`, `.jpeg`, `.png`, `.webp`), unreadable files, and images `< 100x100` px. Flags warnings for low light (`brightness_mean < 40.0`), blur (`Laplacian variance < 100.0`), and overexposure (`brightness_mean > 240.0`). |
| `model.py` | Yes (6,322 B) | Builds `efficientnet_b3` (or `efficientnet_b0` on CPU fast profile) with a 2-class binary classifier head (`CLASS_NAMES = ("non-anemic", "anemic")`). **Does not implement continuous Hb regression.** |
| `predict.py` | Yes (14,190 B) | `AnemiaPredictor` and `PredictionResult`. Runs `validate_image`, transforms image, runs binary classification with `TemperatureScaler`, generates Grad-CAM (`gradcam.py`), and maps binary class + confidence into static text strings via `estimate_hemoglobin_range()` (e.g., `"7-9 g/dL (estimated)"` — heuristic text lookup, **not** a regression model output). |
| `storage_utils.py` | Yes (2,369 B) | Portable path resolution (`uploads/<filename>`, `gradcam/<filename>`) across local `static/` and persistent runtime directories. |
| `pdf_report.py` | Yes (9,861 B) | ReportLab A4 PDF generator (`generate_pdf_report`) rendering header, patient info card, binary verdict banner, confidence bar, original + Grad-CAM images side by side, medical advice, and clinical notes. |
| `gradcam.py` | Yes (13,562 B) | `TemperatureScaler` (reads `models/temperature_scaling.json`) and Grad-CAM heatmap overlay generator (`generate_and_save_gradcam`). |
| `dataset.py`, `train.py`, `evaluate.py` | Yes | Dataset discovery/loaders, training loop, and evaluation script. |
| `schema_patches.py`, `migrate.py` | Yes | Idempotent schema migration helpers (e.g. `scans.deleted_at` soft-delete column). |
| `normalization.py` | **No** | Not present in baseline repository (to be implemented by Person 1). |

### Templates & Static Assets
- **`templates/`:** `base.html`, `index.html`, `scan.html`, `result.html`, `history.html`, `analytics.html`, `about.html`, `author.html`, `404.html`, `500.html`
- **`static/css/`:** `main.css`, `components.css`, `animations.css`, `pages/{index,result,history,analytics,author}.css`
- **`static/js/`:** `main.js`, `upload.js` (client-side drag-and-drop upload, `getUserMedia` rear-camera capture with `facingMode: { ideal: "environment" }`, client-side Laplacian blur/brightness preview, `/health` model-readiness polling), `analytics.js`, `background.js`, `scroll-reveal.js`

### Flask Route Inventory
| Route | Methods | Purpose & Schema |
| :--- | :--- | :--- |
| `/` | `GET` | Homepage dashboard with scan stats (`home_stats()`), 5 most recent scans, and performance overview. |
| `/dashboard` | `GET` | Redirects (`302`) to `/`. |
| `/scan` | `GET` | Patient intake form (`patient_name`*, `age`*, `gender`*, `phone`, `image_type`, `notes`) + file upload / rear-camera capture UI. |
| `/predict` | `POST` | Multipart form submission handler; runs `run_prediction_workflow()` and renders `result.html` (or flashes warning and redirects to `/scan` if model is not loaded). |
| `/api/predict` | `POST` | JSON or multipart inference endpoint. Expects `{image (base64/data-URI), patient_name, age, gender, phone?, image_type?, notes?}`. Returns `{success, id, scan_id, public_scan_id, persisted, patient_name, prediction, confidence, confidence_percent, risk_level, risk_explanation, anemic_probability, non_anemic_probability, hemoglobin_estimate, medical_advice, quality, image_quality, quality_warning, quality_warnings, image_url, gradcam_url, result_url, pdf_url, processing_time_ms, warnings, error}`. |
| `/result/<int:scan_id>` | `GET` | Displays saved scan result page (`result.html`) with original vs. Grad-CAM comparison slider. |
| `/history` | `GET` | Searchable (FTS5), filterable, sortable, paginated scan history view. |
| `/history/export/csv` | `GET` | Downloads filtered scan history as `anemiavision_history.csv`. |
| `/history/delete/<int:scan_id>` | `POST` | Soft-deletes a scan by setting `deleted_at`. |
| `/analytics` | `GET` | Renders Chart.js analytics dashboard from SQLite scan records. |
| `/export/pdf/<int:scan_id>` | `GET` | Generates and downloads `anemiavision_report_<scan_id>.pdf`. |
| `/media/uploads/<path:filename>` | `GET` | Serves uploaded scan image from disk or `MediaFile` SQLite blob fallback. |
| `/media/gradcam/<path:filename>` | `GET` | Serves Grad-CAM overlay image from disk or `MediaFile` SQLite blob fallback. |
| `/about` | `GET` | Displays dataset overview and model performance metrics. |
| `/author` | `GET` | Displays developer profile page. |
| `/health` | `GET` | Readiness JSON endpoint (`status`, `checkpoint_available`, `predictor_loaded`, `predictor_load_error`, `database_backend`, etc.). |

---

## 5. Baseline Test Results (Passed, Failed, Untested)

### Passed Checks (Verified)
- **Application Startup (`python app.py`):** **PASSED** — Starts cleanly on `http://127.0.0.1:5000`, creates `database/anemia_vision.db`, applies schema patches, configures SQLite WAL mode, and initializes the `scan_name_fts` FTS5 virtual table.
- **Health Endpoint (`GET /health`):** **PASSED (HTTP 200)** — Returns JSON health payload (`"status": "ok"`, `"database_backend": "sqlite"`, `"search_index_ready": true`, `"predictor_loaded": false`, `"predictor_load_error": "invalid load key, 'v'."`).
- **Page Routes (`GET /`, `/scan`, `/history`, `/analytics`, `/about`, `/author`, `/dashboard`):** **PASSED (HTTP 200 / 302->200)** — All templates render without error.
- **Image Validator (`image_validator.validate_image`):** **PASSED**
  - Bundled sample image (`static/uploads/20260412_213235_af19d6b1.jpeg`, 224x224): `is_valid=True`, `image_quality="Good"`, `blur_score=118.49`, `brightness_mean=114.34`.
  - Synthetic tiny image (`50x50` JPG): Rejected (`is_valid=False`, `"Image resolution is too small. Please upload an image that is at least 100x100 pixels."`).
  - Synthetic dark/flat image (`200x200` JPG): Accepted with warnings (`is_valid=True`, `image_quality="Poor lighting"`, 2 warnings for low brightness and blur).
  - Unsupported file extension (`.txt`): Rejected (`is_valid=False`, `"Unsupported file format. Please use one of: jpeg, jpg, png, webp."`).
- **SQLite Persistence & FTS5 Search (`Scan`, `scan_name_fts`):** **PASSED** — Verified inserting a `Scan` record (`AV-000001`), querying via FTS5 `patient_name_match_ids()`, rendering `/result/1` (`HTTP 200`), exporting `/history/export/csv` (`HTTP 200`, `text/csv`), and cleaning up the test record.
- **PDF Report Generation (`pdf_report.generate_pdf_report` & `GET /export/pdf/<scan_id>`):** **PASSED** — Generated valid 282,197-byte PDF (`%PDF-1.4`) embedding both original scan and Grad-CAM images, and served via `/export/pdf/1` (`HTTP 200`, `application/pdf`).
- **Media Serving Routes (`GET /media/uploads/...` & `GET /media/gradcam/...`):** **PASSED (HTTP 200)** — Served bundled upload (`11,430` bytes) and Grad-CAM (`124,737` bytes) assets.

### Failed Checks (Verified Blockers)
- **Checkpoint Loading (`model.load_checkpoint` / `eager_load_predictor`):** **FAILED**
  - **Exact Error:** `_pickle.UnpicklingError: invalid load key, 'v'.`
  - **Root Cause:** `models/20260428_2028_best_model.pth` and `models/20260428_2028_last_checkpoint.pth` are 134-byte Git LFS text pointer files (`version https://git-lfs.github.com/spec/v1 ... size 140300969`) because `RAHULPATEL2002/anemia-detection` exceeded its GitHub LFS bandwidth quota (`batch response: This repository exceeded its LFS budget`).
- **Live Prediction Endpoint (`POST /api/predict` & `POST /predict`):** **FAILED (HTTP 503)**
  - **Exact Response:** `503 Service Unavailable` — `{"error": "The AI model is still starting on this server. Please wait a few seconds and try again.", "success": false}`
  - **Root Cause:** `latest_available_checkpoint()` only checks `checkpoint_path.exists()` (which is `True` for the 134-byte LFS pointer file), while `predictor_loaded()` remains `False` because `get_predictor()` fails with `UnpicklingError: invalid load key, 'v'.`. Consequently, `/api/predict` assumes the model is still warming up and returns HTTP 503 indefinitely.

### Untested Items
- **Interactive Mobile Rear-Camera Hardware Capture (`navigator.mediaDevices.getUserMedia`):** **UNTESTED** — Verified in `templates/scan.html` and `static/js/upload.js`, but requires physical camera hardware and a browser context (`localhost` or HTTPS) to exercise live video capture.
- **PostgreSQL Backend:** **UNTESTED** — Local baseline uses SQLite (`database/anemia_vision.db`); no `DATABASE_URL` Postgres instance is configured locally.

---

## 6. File Ownership & Conflict Matrix (Person 1, Person 2, Person 3)

| Component / Path | Primary Owner | Conflict Risk & Integration Rule |
| :--- | :--- | :--- |
| `normalization.py` (new), dataset manifests/splits | **Person 1** | Person 1 owns ROI crop, glare masking, Grey-World white balance, HSV+CLAHE, ITA calculation, and colorimetric feature extraction. Person 3 calls `normalize_image(image, body_part)` via service adapter. |
| `model.py`, `train.py`, `evaluate.py`, `models/*` | **Person 2** | Person 2 owns dual-head (classification + continuous Hb regression) architecture, checkpoint creation, calibration, and evaluation metrics. Person 3 must not edit `model.py` or `train.py`. |
| `predict.py`, `gradcam.py` | **Shared (Person 2 & Person 3)** | **High conflict risk.** Coordinate changes: Person 2 exposes model inference contract; Person 3 wraps preprocessing + inference behind a clean service adapter so route handlers in `app.py` remain decoupled. |
| `app.py`, `pdf_report.py`, `storage_utils.py`, `templates/*`, `static/*` | **Person 3** | Person 3 owns Flask routes, service adapter, CHW triage (Green/Yellow/Red), SQLite persistence updates, referral PDF generation, and UI templates/assets. |

---

## 7. Recommended Next Steps for Person 3, Phase 2 (H2–H4)

1. **Resolve Checkpoint Blocker with Person 2:**
   - Coordinate with Person 2 to provide a valid PyTorch `.pth` checkpoint (either a newly trained checkpoint or an initialized baseline `efficientnet_b3` checkpoint saved via `model.save_checkpoint()`) to replace the 134-byte LFS pointer in `models/`.
   - In Phase 2/3, add an LFS-pointer/corrupt-checkpoint guard in the service adapter so `app.py` surfaces `_predictor_load_error` clearly instead of reporting `"The AI model is still starting"`.
2. **Lock Interface Contracts (H2–H4):**
   - **With Person 1 (`normalization.py`):** Confirm exact signature `normalize_image(image, body_part)` -> `(normalized_image, feature_vector, metadata)` including preview image saving paths, ITA value/category, and quality warning fields.
   - **With Person 2 (`predict.py` / `model.py`):** Confirm JSON/dict output contract: `hb_estimate_g_dl` (nullable float from regression head — replacing the baseline's heuristic string range), `anemia_probability`, `predicted_class`, `model_version`, `calibration_status`, `gradcam_path`, and `warnings`/`error`.
3. **Prepare Service Adapter & Mock Fallback (H3–H8):**
   - Design the backend adapter layer that chains `image_validator.py` -> Person 1's `normalization.py` -> Person 2's predictor, with any development mock explicitly labelled as `MOCK / PROTOTYPE FALLBACK`.
