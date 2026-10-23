# VYRA Integration Handoff (Person 3, branch `Integration`)

Status: **partially integrated. The end-to-end integration test has NOT passed.**

## What is in this branch
- `origin/main` (Person 2: model, predictor, evaluation, docs) merged into the app root.
- `origin/vikas` (Person 1: `person1/` preprocessing package) added.
- Person 3 work from Phases 1-5 is in `person3_app/` (a plain-source copy of the local `anemia-detection/` repo, which has its own `.git`): `service_adapter.py`, updated `app.py`, `model.py`, `predict.py`, `pdf_report.py`, templates, static files, tests, and the `PERSON3_PHASE*.md` reports.
- `PERSON_1_PREPROCESSING_HANDOFF.md` and `PERSON_3_INTEGRATION_SPEC.md` are also in `docs/`.

## Verified
- `person1/tests`: 65 passed. This needed a fix in `person1/src/vyra_preprocessing/config.py`. `pipeline_fingerprint` crashed because `opencv-python-headless` was not installed as a distribution. It now falls back to `cv2.__version__`.
- `tests/test_model_unit.py` and `tests/test_vyra_pipeline.py`: 18 passed. This was only after I added a placeholder checkpoint and synthetic sample images (see below).
- Phases 3-5 tests (30 in total) passed earlier against the old LFS-pointer state. They have **not** been re-run against the merged code.

## NOT done / known problems
1. **No real model checkpoint.** `checkpoints/*.pth` and `models/*.pth` are gitignored in `origin/main`, so Person 2's real `best_dual_head.pth` was never in the repo. The local `models/20260428_2028_*.pth` files are 134-byte Git LFS pointers.
   - `scripts/build_verified_checkpoint.py` (new, mine) generates a **placeholder** checkpoint. It trains only the fusion and heads on the committed Person 1 feature manifests, with a random, untrained visual backbone.
   - It is **not** Person 2's champion model. Its validation accuracy was about 50-57%, and it only exists to make the 18 tests pass.
   - The `.pth` files and synthetic `data/raw/...` images it creates are gitignored and not pushed.
   - **Person 2 must supply the real `best_dual_head.pth`** (the Hugging Face release, a Git LFS file, or a direct share). It has to carry `model_state_dict`, `temperature` (1.3391530513763428) and `has_validated_regression: True`.
2. **The Flask app is not wired to the new predictor.** The Flask `app.py` still loads the legacy `AnemiaPredictor` (EfficientNet-B0/B3 binary) through `model.py` and `config.py`. These still point at `models/20260428_2028_best_model.pth`.
   - They need to switch to `checkpoints/best_dual_head.pth`, and `service_adapter.py` needs to call `src.inference.predictor.predict`.
   - `predict.py` and `gradcam.py` assume a single-output model. Grad-CAM with `DualHeadEfficientNetB3` needs the model output tuple handled (`logits, hb = model(...)`) and a 3-input forward pass. My attempted `gradcam.py` edit did **not** apply.
3. **Two normalization paths conflict.** Person 2's `src/data/normalization_adapter.py` accepts any full-frame image. Person 1's `vyra_preprocessing.preprocess_image` rejects opaque full-frame photos unless an ROI or `pre_cropped=True` is given. Person 1 and Person 2 need to agree which one production uses. Training and inference must use the same one.
4. `app.py` `performance_overview()` does not yet read `evaluation/metrics/champion_test_report.json`.
5. `tests/test_phase5_e2e_verification.py` and `test_phase4` assert the LFS-pointer or "predictor missing" state. They will need updating once the real model is wired.
6. The integration test (upload, prediction, results page, PDF) has not been run against the real model.

## Honest model performance (Person 2's held-out test report, n=1025)
- Accuracy 63.9%, sensitivity 67.4%, specificity 58.8%, ROC-AUC 0.664.
- Hb MAE 2.06 g/dL, limits of agreement about [-5.0, +5.4] g/dL.
- Present this as a screening prototype. It does not replace a laboratory CBC.
- Fingernail output must have `hb_estimate_g_dl = null` and `regression_status = "unavailable"`.

## Needed from teammates
- **Person 2:** the real `best_dual_head.pth` (and `models/best_dual_head.pth`), plus one sample conjunctiva image and one fingernail image for smoke tests.
- **Person 1 and 2:** a decision on the single normalization module for live uploads (ROI handling).

## Next steps for whoever continues
1. Place the real checkpoint at `checkpoints/best_dual_head.pth` and `models/best_dual_head.pth`.
2. Point `config.py` and `model.py` at it, and route `service_adapter.run_vyra_predict` to `src.inference.predictor.predict`.
3. Fix `gradcam.py` for the dual-head model, and add the normalized image and ITA fields to the result page and PDF.
4. Update the Phase 4/5 tests and run `pytest tests person1/tests anemia-detection/tests`.
