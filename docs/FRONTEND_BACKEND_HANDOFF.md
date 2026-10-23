# VYRA React frontend → Integration backend handoff

## Delivery and responsibility

This is a **frontend-only integration** against Integration commit
`67b3b6909965411020dc662a0e86b0fc2ca5938b`. The existing React site, local fonts,
eye film, anatomical models and UI tests are under `web/`. Python prediction,
ROI, explanation, stress-test, PDF and database implementations are unchanged.
The existing local workspace/history were not migrated.

The UI's new-feature browser tests use fixtures identified as test data. Passing
those tests establishes UI behavior, **not** live backend or clinical validation.
The backend developer must complete the connection and acceptance checks below.
No dataset is needed to build the frontend. Real inference needs the trained
checkpoint, preprocessing/detector dependencies and the backend's runtime files.

### Files and runtime on the backend computer

- Reuse `checkpoints/best_dual_head.pth`, or the backend's configured
  `ANEMIA_BEST_CHECKPOINT` path. The supplied champion is 47,314,074 bytes with
  SHA256 `57b6a4be8b14d11c2574c64192983942ae3d6ba760ddc3fca0828341f5d28157`.
  Its reported embedded temperature is `1.3391530513763428`; reconcile any older
  standalone JSON before claiming checkpoint parity. Do not substitute weights.
- Keep the detector's `models/face_detection_yunet.onnx` and required Python
  dependencies available. Report actual detector/model readiness to the UI.
- Preserve the existing database and upload/report directories. Use a disposable
  `ANEMIA_RUNTIME_ROOT` and test database for acceptance runs; explicit
  `DATABASE_URL`/`SQLALCHEMY_DATABASE_URI` settings take precedence and must also
  point to the intended test database.
- For live explanations, provision `qwen2.5:7b` in Ollama and configure
  `OLLAMA_BASE_URL` on the backend. These are backend settings, never frontend
  secrets or a reason to call Ollama directly from the browser.
- Raw training datasets are not frontend build inputs. EDA routes should return
  available metadata/figures or explicit missing-data states without inventing
  samples.

## Build and serve

Requirements: Node.js 22.12+ and npm. On the backend computer:

```sh
cd web
npm ci
npm test
npm run build:integration
cd ..
python app.py
```

The build is written only to `person3_app/static/ui/`. Open
`http://127.0.0.1:5000/static/ui/index.html` and use the hash routes below. It uses
Flask's existing static serving; **do not replace any legacy template handlers**.
A final deployment may add a `/ui/` redirect to this entry page, but it is optional.
Keep this static prefix when serving assets; the GLBs, film, fonts and lazy chunks
use it. The source `web/` is canonical; rebuild rather than hand-editing output.
If transferring the frontend ZIP instead of the Git commit, add
`/person3_app/static/ui/` to the target repository `.gitignore` before building.
The Git commit already includes that ignore rule.

For hot reload, copy `web/.env.example` to `web/.env.local`, configure
`VYRA_BACKEND_URL`, then run `npm run dev` inside `web/`. This is a development
proxy setting, not a browser-exposed secret or a cloud inference endpoint.
Production requests are same-origin. Do not broadly enable CORS for deployment.

Hash routes: `/`, `/studio`, `/studio/jobs/:id`, `/history`, `/history/:id`,
`/anatomy`, `/evidence`, `/analytics`, `/research`, `/achilles-heel`, `/about`.
Legacy `/`, `/scan`, `/history`, `/analytics`, `/eda`, `/achilles-heel`, `/about`,
`/predict`, `/export/pdf/:scan_id` continue to belong to the Flask application.

## Backend changes the backend developer must implement

The current Integration application does **not** expose the React application's
job/history contract. Merely serving the assets or pointing the proxy at port 5000
will not connect screening. Add an isolated compatibility blueprint under
`/api/v1`, reusing the existing prediction and persistence functions. Do not
replace the backend with the old `src/vyra` application, load a second model, or
create a second screening history. Preserve existing routes, request contracts,
`Scan` columns and current model/calibration/preprocessing behavior.

### Readiness and capability discovery

`GET /api/v1/system` must return:

```json
{
  "api_version": "1",
  "patient_intake_version": 1,
  "ready": true,
  "model": {
    "name": "checkpoint model name",
    "sha256": "actual SHA256",
    "temperature": 1.3391530513763428,
    "regression_supported": true
  },
  "detector": { "available": true },
  "limits": { "max_bytes": 10485760, "min_side": 50 },
  "warnings": [],
  "capabilities": {
    "dual_scan": true,
    "assistant": true,
    "stress_test": true,
    "colorimetry": true,
    "research": true
  }
}
```

These are shape examples, not expected readiness results. Report actual limits,
readiness, hashes and calibration. Only advertise implemented capabilities.
`assistant:true` means the explanation route is supported, including its real
server fallback; it does not mean Ollama is currently running. Absent capabilities
are treated as unavailable. Missing `patient_intake_version:1` blocks uploads so
an older backend cannot silently discard fields.

### Jobs and patient fields

| Method and URL                  | Required behavior                                                                                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /api/v1/jobs`             | Multipart upload; return Job JSON and 202. Accept `image`, `modality`, `reference`, `age`, `sex`, optional `phone`, `notes`, and optional `secondary_image`. Honor `Idempotency-Key`. |
| `GET /api/v1/jobs/:id`          | Restore the requested job and actual persisted progress.                                                                                                                              |
| `POST /api/v1/jobs/:id/confirm` | Accept `{"roi":Box}` or `{"roi":Box,"secondary_roi":Box}`; queue actual inference only after review.                                                                                  |
| `POST /api/v1/jobs/:id/cancel`  | Cancel pending work without saving a fabricated/completed result.                                                                                                                     |

Map `reference` to `patient_name`, `sex` to the backend gender representation,
and canonical modality (`conjunctiva`/`fingernail`) to the existing adapter's body
part. Support explicit `not-stated` without pretending it is a different known
sex. Patient details remain record context unless a separately trained model
explicitly supports them. Never discard phone/notes silently.

`Box` is an integer `{x0,y0,x1,y1}`, exclusive right/bottom, in the server-oriented
image coordinates. Detection proposes a box; review confirms it. Use the **same
confirmed tissue** for inference, color measurements and displayed artifacts.
Validate boxes on the server independently of client checks. For dual capture,
the UI currently fixes eye as primary and nail as secondary. Both require review.

Job JSON follows `web/src/types.ts` and includes `id`, `patient`, `status`, `stage`,
`modality`, timestamps, elapsed_seconds, `image:{url,width,height}|null`, detection,
roi, warnings, stages, scan_id and error. Statuses: `queued`, `detecting`,
`awaiting_roi`, `running`, `completed`, `failed`, `cancelled`, `interrupted`.
Stages: validation, detection, roi, normalization, features, inference, explanation,
saving. Report stages from actual server work, never timer-based simulation.

For dual jobs include `dual_scan:true` and:

```json
{
  "secondary": {
    "modality": "fingernail",
    "image": {
      "url": "/media/uploads/example.png",
      "width": 640,
      "height": 480
    },
    "detection": {
      "success": true,
      "roi": { "x0": 50, "y0": 50, "x1": 300, "y1": 300 },
      "type": "actual_detector",
      "confidence": 0.9,
      "message": "Review the proposed region",
      "retake_guidance": null
    },
    "roi": { "x0": 50, "y0": 50, "x1": 300, "y1": 300 },
    "error": null
  }
}
```

A missing/failed secondary review must remain explicit. Do not silently convert a
dual request to single capture or present fusion after secondary inference fails.
The existing `/api/predict` currently fails to forward `secondary_image`, whereas
`/predict` forwards it. This must be accounted for in the new job adapter.

If durable job state needs storage, keep it in a separate job store or approved
sidecar, so the existing `Scan` database schema is not changed by this integration.
Serialize inference/Grad-CAM/stress work safely around shared model hooks. Do not
run a second copy of the inference backend alongside the canonical one.

### Saved results, history, evidence and analytics

Add JSON endpoints compatible with the TypeScript interfaces:

- `GET /api/v1/scans` → `{"scans": Scan[]}`; active records only.
- `GET /api/v1/scans/:id` → Scan. Normalize numeric database IDs to JSON strings;
  retain the actual numeric value in route calls to the existing backend.
- `DELETE /api/v1/scans/:id` → 204, mapped to Integration's **archive/soft-delete**
  behavior. Do not change the backend's retention policy or claim images were
  permanently erased by archiving.
- `GET /api/v1/scans.csv` → actual active history CSV.
- `GET /api/v1/analytics` → `Analytics` in `web/src/types.ts`, derived from active
  records. Zero records means zeros/null means, never demonstration statistics.
- `GET /api/v1/evidence` → actual supplied champion evaluation/provenance matching
  `Evidence` in the same file. Do not replace reported evaluation with new claims.
- `GET /api/v1/samples/:modality` → permitted real reference image or an explicit
  unavailable response. Do not return an anatomical illustration as patient input.

A Scan includes existing model probability/threshold as **fractions 0–1**, eye Hb
or null, regression status, original patient context, model/preprocessing identity,
ROI/image dimensions, warnings, features, timings, follow-up and artifacts.
Map normalized and Grad-CAM URLs explicitly; do not invent missing artifacts.
Expose a safe, same-origin `pdf_url` pointing to `/export/pdf/<numeric_id>`.
The exact existing UI record interface is in `web/src/types.ts`.

Keep image-estimated Hb distinct from laboratory Hb; nail Hb remains unavailable.
A dual result may show the eye-derived Hb, but must not invent a fused/nail Hb.
Follow-up guidance must have an explicit source/version and must not be silently
replaced by an unrelated frontend recommendation generator.

### Fusion and colour measurements

Expose these as optional **structured** JSON fields on the Scan response;
no Scan column additions are needed:

- `multimodal_fusion`: valid decoded payload from the exact
  `MULTIMODAL_FUSION:` envelope. Existing keys are `active`, `primary_modality`,
  `primary_prob`, `primary_class`, `secondary_modality`, `secondary_prob`,
  `secondary_class`, `joint_probability`, `concordance_pct`, `agreement_level`.
  All four probability/agreement numbers in this object are **percent 0–100**.
  Optionally provide `secondary_image_url`, `method` and `secondary_error`.
- Return **patient notes separately from machine metadata**. Do not show the
  envelope JSON as clinical notes or treat arbitrary marker text as trusted JSON.
- `biophysical_metrics`: `{has_real_pixels,erythema_index,l_star,a_star,b_star,source?}`.
  Return null for absent measurements. The frontend suppresses values unless
  `has_real_pixels:true` and all displayed numbers are finite.

Important findings for the backend developer to review, not frontend repairs:

1. The pinned fusion implementation is `0.55*eye + 0.45*nail`, **not Bayesian
   posterior fusion**. Concordance is `100*(1-abs(p_eye-p_nail))`; it is not a
   diagnostic confidence or a fixed88.5–93.8% range. Preserve calculations unless
   intentionally revalidated, and describe them accurately.
2. `Scan.biophysical_metrics` currently fabricates example numbers when image
   files are absent or undecodable. Return unavailable values for those cases;
   `has_real_pixels:false` is not permission to display placeholders as findings.
3. EI currently includes a +0.16 offset and 5–95 clipping and uses the stored image.
   Document the actual formula, colour space and source crop. Ensure it measures
   the intended tissue rather than facial/background pixels. UI terminology is
   colourimetry; RGB/Lab do not establish a measured542/576 nm spectrum or blood
   oxygen/perfusion percentage.
4. The8-bit Lab conversion must retain physical units (`L* = encodedL*100/255`,
   `a* = encodedA-128`, `b* = encodedB-128`).

### Grad-CAM

Provide `artifacts.normalized` and `artifacts.gradcam` for the same crop.
Optional `artifacts.heatmap` can carry a raw colour heatmap, aligned to that crop.
With the raw heatmap, blend opacity can be controlled directly; with only the
existing precomposited PNG, the UI accurately labels a cross-fade of the overlay.
Neither display is evidence that the model's prediction is correct.

### Explanation assistant — existing route preserved

`POST /api/v1/explain/<numeric_scan_id>` with:

```json
{ "question": "What does my image specifically show?", "language": "en" }
```

Response contract:

```json
{
  "is_emergency": false,
  "response": "Grounded response text",
  "model_used": "deterministic_guardrail",
  "grounded_facts": {}
}
```

`model_used` is `qwen2.5:7b`, `deterministic_guardrail`, or `emergency_firewall`.
Emergency responses include `rule` and `hotlines`. Both `en` and `hi` must work.
The frontend displays the actual source and renders text/bullets/bold safely;
it does not generate medical answers when the API fails. A404/timeout is an
unavailable state, not a successful local-model/fallback response.

On the GPU computer, configure Ollama at the backend's local address, provision
`qwen2.5:7b`, and verify availability, GPU use, keep_alive and timeout behavior.
This computer had neither Ollama nor NVIDIA tooling installed. Live Qwen remains
an explicit backend acceptance item. Review emergency wording, negation handling,
English/Hindi matching, prohibited dosage/transfusion advice and generated-text
guardrails with appropriately qualified reviewers. Certification is not
established by this source handoff; do not describe fallback text as certified.

### Achilles Heel — existing route preserved

`POST /api/v1/achilles-heel` accepts multipart `image`, `modality`. Its response is
`StressReport` in `web/src/integration.ts`. **Mixed units matter:**
`baseline_probability` is percent 0–100; `conditions[].probability` and
`delta_prob` are fractions 0–1; `percent_shift`, mechanism sensitivity and the
overall index are percent 0–100. These scores describe this experiment, not
clinical accuracy or calibrated failure probabilities.

The SPA can accept `?scan_id=<id>` in its hash route, load the saved source image
and submit that image to the unchanged endpoint. Preserve the legacy link
`/achilles-heel?scan_id=<id>` as well. Results must not overwrite the original
screening or appear in clinical history as another patient screening.

Replace hard-coded Windows example paths and the neutral-grey fallback with
permitted bundled examples or an explicit unavailable state. A missing source
image must never silently run the battery on a grey image. Validate uploads and
resource limits. Warm/cool conditions are simulated colour transformations, not
spectrometer measurements or proven causal diagnoses of model bias.

### Research/EDA JSON

Add `GET /api/v1/research` returning:

```json
{
  "summary": "Description of the actual supplied research assets",
  "datasets": [
    {
      "name": "Dataset name",
      "description": "Provenance and scope",
      "count": null
    }
  ],
  "plots": [
    {
      "title": "Plot title",
      "url": "/static/plots/actual.png",
      "description": "What was evaluated"
    }
  ]
}
```

Only expose data/plots actually available and permitted to display. Do not require
raw datasets just to start the website. Legacy `/eda` remains available.

## Backend acceptance checklist — pending on backend computer

1. Record baseline `pytest person3_app/tests/ tests/`; preserve all existing tests.
2. Verify `python app.py` starts on 5000 with the intended database and checkpoint.
3. Verify single eye and single nail through both legacy routes and the React UI;
   check sample outputs against the supplied reference tolerances and null nailHb.
4. Verify real dual upload, both crop confirmations, partial failure, note
   preservation, fraction/percent formatting and correct PDF record ID.
5. Reload an active job; test retry/idempotency, cancellation, restart and archive.
6. Verify normalized/Grad-CAM alignment and blend/split, real EI/Lab, missing-image
   unavailability and legacy scans without new fields.
7. Ask "Hi" and a scan-specific question in English and Hindi with real Qwen;
   repeat with Ollama stopped/timed out. Verify emergency handling on both paths.
8. Run all 8 stress conditions on actual eye/nail images; missing/invalid data must
   fail explicitly and must not change the saved result.
9. Download the actual PDF; verify patient details, data provenance, eye-onlyHb,
   report wording, archive retention and source measurements.
10. Run frontend checks and record backend/model/LLM versions plus which checks
    were actually executed. Do not relabel fixture tests as live verification.

## Compatibility hooks

The React controls retain the requested IDs: cameraVideo, cameraCanvas,
targetingReticle, lightingBadge, focusBadge, shutterBtn, primaryDropzone,
secondaryDropzone, explainabilityViewer, gradcamSlider, btnModeBlend,
btnModeSplit, erythemaFillBar, downloadPdfButton, assistantSection,
assistantChatStream, assistantInput, assistantSendBtn and assistantChips.
Language buttons use `.lang-btn[data-lang=en|hi]`. React owns events on the SPA;
do not include legacy upload/result scripts on that entry page. Camera telemetry
is a local brightness/edge-detail heuristic sampled every 280 ms, not a tissue
detector or validated clinical quality gate. Tracks/timers stop on close,
unmount, hidden tabs and disconnection; reduced motion suppresses the shutter
sweep. Static UI output is generated and ignored by Git; build it on the backend
computer before opening the static entry URL.

## What this phase does not establish

No backend/model/clinical changes, raw-dataset evaluation, hospital certification,
new accuracy claim, government integration, production deployment, patient data
migration, live GPU/Qwen verification or cloud account setup is included.
