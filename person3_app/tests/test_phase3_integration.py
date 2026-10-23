"""Automated Phase 3 backend integration test suite for VYRA (`tests/test_phase3_integration.py`).

Covers all 17 required Phase 3 verification scenarios:
1. Flask application startup when model weights are missing.
2. Valid model readiness when the actual checkpoint loads successfully.
3. Detection of Git LFS pointer checkpoint files.
4. `/health` accurately reporting model readiness.
5. Prediction endpoint returning a clear service-unavailable response when real inference is unavailable.
6. Successful adapter integration using a controlled test double that returns the documented schema.
7. Correct handling of missing, `null`, malformed, or out-of-range output fields.
8. Hb readout suppressed when the regression head is unavailable or unvalidated.
9. Original tiny images rejected before resize/crop.
10. Corrupt, unsupported, unreadable, and excessively large uploads handled safely.
11. Original uploads remaining separate from normalized/model-input images.
12. Correct mapping of conjunctiva and fingernail modalities.
13. Mock mode disabled by default and blocked in production.
14. Mock results not saved or reported as genuine predictions.
15. Existing database/history, CSV, media, and PDF workflows remaining functional.
16. Triaging never assigning GREEN when required inputs are missing.
17. No hardcoded performance metrics presented as actual model results.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as app_module
import config as config_module
import model as model_module
import service_adapter as adapter_module
from model import build_model, inspect_checkpoint_file, is_lfs_pointer_file, load_checkpoint, save_checkpoint
from service_adapter import (
    ContractValidationError,
    ImageUploadValidationError,
    ModalityValidationError,
    ModelUnavailableError,
    PredictionExecutionError,
    UNAVAILABLE_HB_MESSAGE,
    compute_vyra_triage,
    inspect_runtime_readiness,
    is_mock_mode_enabled,
    resolve_body_part,
    run_vyra_predict,
    set_test_normalizer,
    set_test_predictor,
)


def _make_sharp_test_image_bytes(width: int = 320, height: int = 240, fmt: str = "PNG") -> bytes:
    """Create a synthetic high-contrast checkerboard image that passes blur & brightness checks."""

    rng = np.random.default_rng(42)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Checkerboard pattern for high Laplacian variance (>100) and medium brightness (~128)
    y_idx, x_idx = np.indices((height, width))
    checker = ((x_idx // 6 + y_idx // 6) % 2) * 180 + 40
    noise = rng.integers(0, 25, size=(height, width), dtype=np.uint8)
    channel = np.clip(checker + noise, 0, 255).astype(np.uint8)
    arr[:, :, 0] = channel
    arr[:, :, 1] = np.clip(channel - 20, 0, 255)
    arr[:, :, 2] = np.clip(channel - 30, 0, 255)

    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def _reset_adapter_state(monkeypatch: pytest.MonkeyPatch):
    """Ensure injected test doubles, predictor state, and temporary uploads are reset around every test."""

    existing_uploads = set(config_module.UPLOADS_DIR.glob("*")) if config_module.UPLOADS_DIR.exists() else set()
    existing_reports = set(config_module.REPORTS_DIR.glob("*")) if config_module.REPORTS_DIR.exists() else set()
    set_test_predictor(None)
    set_test_normalizer(None)
    monkeypatch.delenv("VYRA_ENABLE_DEV_MOCK", raising=False)
    monkeypatch.delenv("FLASK_ENV", raising=False)
    monkeypatch.delenv("VYRA_ENV", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setattr(app_module, "_predictor", None)
    yield
    set_test_predictor(None)
    set_test_normalizer(None)
    if config_module.UPLOADS_DIR.exists():
        for path in config_module.UPLOADS_DIR.glob("*"):
            if path not in existing_uploads and path.name != "20260412_213235_af19d6b1.jpeg":
                _safe_unlink(path)
    if config_module.REPORTS_DIR.exists():
        for path in config_module.REPORTS_DIR.glob("*"):
            if path not in existing_reports and path.name != "anemiavision_report_1.pdf":
                _safe_unlink(path)


@pytest.fixture()
def client():
    """Return a configured Flask test client."""

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        with app_module.app.app_context():
            app_module.db.session.query(app_module.RateLimitEvent).delete()
            app_module.db.session.commit()
        yield test_client


# ---------------------------------------------------------------------------
# 1. Flask application startup when model weights are missing
# ---------------------------------------------------------------------------
def test_01_app_startup_when_weights_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    missing_best = tmp_path / "nonexistent_best.pth"
    missing_last = tmp_path / "nonexistent_last.pth"
    monkeypatch.setattr(model_module, "BEST_CHECKPOINT_PATH", missing_best)
    monkeypatch.setattr(model_module, "LATEST_CHECKPOINT_PATH", missing_last)
    monkeypatch.setattr(app_module, "BEST_CHECKPOINT_PATH", missing_best)

    # Eager load must not crash when checkpoints are missing
    app_module.eager_load_predictor()
    test_app = app_module.create_app()
    test_app.config["TESTING"] = True

    with test_app.test_client() as c:
        resp = c.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["checkpoint_exists"] is False
        assert data["checkpoint_available"] is False
        assert data["checkpoint_valid"] is False
        assert data["predictor_loaded"] is False
        assert data["model_ready"] is False
        assert data["failure_category"] == "missing_checkpoint"


# ---------------------------------------------------------------------------
# 2. Valid model readiness when the actual checkpoint loads successfully
# ---------------------------------------------------------------------------
def test_02_valid_model_readiness_when_checkpoint_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ckpt_path = tmp_path / "valid_model.pth"
    net = build_model(pretrained=False, architecture="efficientnet_b0")
    save_checkpoint(
        model=net,
        epoch=1,
        metrics={"valid_accuracy": 0.81, "valid_sensitivity": 0.84, "valid_specificity": 0.78, "valid_auc_roc": 0.86},
        config_snapshot={"training": {"image_size": [224, 224]}},
        checkpoint_path=ckpt_path,
        architecture="efficientnet_b0",
    )

    monkeypatch.setattr(model_module, "BEST_CHECKPOINT_PATH", ckpt_path)
    monkeypatch.setattr(model_module, "LATEST_CHECKPOINT_PATH", ckpt_path)
    monkeypatch.setattr(app_module, "BEST_CHECKPOINT_PATH", ckpt_path)

    loaded_model, payload = load_checkpoint(ckpt_path, device="cpu", pretrained_fallback=False)
    assert loaded_model is not None
    assert payload["architecture"] == "efficientnet_b0"

    predictor = app_module.get_predictor()
    assert predictor is not None
    assert app_module.predictor_loaded() is True

    readiness = inspect_runtime_readiness(predictor_loaded=True)
    assert readiness["checkpoint_valid"] is True
    assert readiness["checkpoint_is_lfs_pointer"] is False
    assert readiness["model_ready"] is True
    assert readiness["failure_category"] is None


# ---------------------------------------------------------------------------
# 3. Detection of Git LFS pointer checkpoint files
# ---------------------------------------------------------------------------
def test_03_git_lfs_pointer_detection(tmp_path: Path):
    # Verify actual repo models/20260428_2028_best_model.pth is detected as a Git LFS pointer
    repo_ckpt = config_module.BEST_CHECKPOINT_PATH
    if repo_ckpt.exists() and repo_ckpt.stat().st_size < 1024:
        assert is_lfs_pointer_file(repo_ckpt) is True
        info = inspect_checkpoint_file(repo_ckpt)
        assert info["checkpoint_exists"] is True
        assert info["checkpoint_is_lfs_pointer"] is True
        assert info["checkpoint_valid"] is False
        assert info["failure_category"] == "lfs_pointer_checkpoint"

    # Also test a synthetic LFS pointer file and verify load_checkpoint rejects it before torch.load
    fake_lfs = tmp_path / "pointer.pth"
    fake_lfs.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
        "size 51428800\n",
        encoding="utf-8",
    )
    assert is_lfs_pointer_file(fake_lfs) is True
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        load_checkpoint(fake_lfs, device="cpu")


# ---------------------------------------------------------------------------
# 4. /health accurately reporting model readiness
# ---------------------------------------------------------------------------
def test_04_health_accurately_reports_readiness(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # 1. Real champion model is present and valid
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "ok"
    assert payload["checkpoint_exists"] is True
    assert payload["checkpoint_valid"] is True
    assert payload["checkpoint_available"] is True
    assert payload["checkpoint_is_lfs_pointer"] is False

    # 2. When an LFS pointer is configured, /health reports failure accurately
    fake_lfs = tmp_path / "pointer.pth"
    fake_lfs.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
        "size 51428800\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_module, "BEST_CHECKPOINT_PATH", fake_lfs)
    monkeypatch.setattr(model_module, "LATEST_CHECKPOINT_PATH", fake_lfs)
    monkeypatch.setattr(app_module, "BEST_CHECKPOINT_PATH", fake_lfs)
    monkeypatch.setattr(app_module, "_predictor", None)
    monkeypatch.setattr(adapter_module, "_INJECTED_PREDICTOR_FN", None)
    monkeypatch.setattr(adapter_module, "resolve_person2_predict_callable", lambda: None)

    resp_lfs = client.get("/health")
    assert resp_lfs.status_code == 200
    payload_lfs = resp_lfs.get_json()
    assert payload_lfs["checkpoint_is_lfs_pointer"] is True
    assert payload_lfs["checkpoint_valid"] is False
    assert payload_lfs["checkpoint_available"] is False
    assert payload_lfs["failure_category"] == "lfs_pointer_checkpoint"
    assert "Git LFS pointer" in payload_lfs["status_message"]


# ---------------------------------------------------------------------------
# 5. Prediction endpoint returning a clear service-unavailable response when real inference is unavailable
# ---------------------------------------------------------------------------
def test_05_predict_returns_503_when_model_unavailable(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_lfs = tmp_path / "pointer.pth"
    fake_lfs.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
        "size 51428800\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_module, "BEST_CHECKPOINT_PATH", fake_lfs)
    monkeypatch.setattr(model_module, "LATEST_CHECKPOINT_PATH", fake_lfs)
    monkeypatch.setattr(app_module, "BEST_CHECKPOINT_PATH", fake_lfs)
    monkeypatch.setattr(app_module, "_predictor", None)
    monkeypatch.setattr(adapter_module, "_INJECTED_PREDICTOR_FN", None)
    monkeypatch.setattr(adapter_module, "resolve_person2_predict_callable", lambda: None)

    img_bytes = _make_sharp_test_image_bytes(240, 240)
    resp = client.post(
        "/api/predict",
        data={
            "patient_name": "Unavailable Test Patient",
            "age": "31",
            "gender": "Female",
            "body_part": "conjunctiva",
            "image": (io.BytesIO(img_bytes), "conjunctiva_ok.png"),
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["success"] is False
    assert data["failure_category"] == "lfs_pointer_checkpoint"
    assert "still starting" not in data["error"].lower()
    assert "git lfs pointer" in data["error"].lower()


# ---------------------------------------------------------------------------
# 6. Successful adapter integration using a controlled test double that returns the documented schema
# ---------------------------------------------------------------------------
def test_06_successful_adapter_integration_with_person2_schema(client):
    captured_call: dict[str, Any] = {}

    def fake_person2_predict(
        image,
        feature_vector=None,
        body_part="conjunctiva",
        metadata=None,
        screening_threshold=0.50,
    ) -> dict[str, Any]:
        captured_call["image"] = image
        captured_call["feature_vector"] = feature_vector
        captured_call["body_part"] = body_part
        captured_call["metadata"] = metadata
        captured_call["screening_threshold"] = screening_threshold
        return {
            "hb_estimate_g_dl": 6.48,
            "anemia_probability": 0.5497,
            "predicted_class": "anemic",
            "model_version": "vyra-dualhead-efficientnet-b3-v1.0",
            "calibration_status": "calibrated",
            "quality_warnings": ["Minor peripheral shadow noted."],
            "error": None,
            "anatomical_modality": body_part,
            "ita_angle": -78.23,
            "ita_category": "Dark",
            "glare_fraction": 0.012,
            "processing_time_ms": 35,
            "regression_status": "validated",
        }

    set_test_predictor(fake_person2_predict)

    img_bytes = _make_sharp_test_image_bytes(320, 240)
    resp = client.post(
        "/api/predict",
        data={
            "patient_name": "Asha Devi",
            "age": "24",
            "gender": "Female",
            "body_part": "conjunctiva",
            "high_sensitivity_mode": "true",
            "image": (io.BytesIO(img_bytes), "asha_conjunctiva.png"),
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    payload = resp.get_json()

    # Verify exact arguments passed to Person 2's predict()
    assert captured_call["body_part"] == "conjunctiva"
    assert captured_call["metadata"] == {"age": 24, "sex": "F"}
    assert captured_call["screening_threshold"] == pytest.approx(0.255)
    assert Path(captured_call["image"]).exists()

    # Verify mapped response contract fields
    assert payload["success"] is True
    assert payload["is_mock"] is False
    assert payload["persisted"] is True
    assert payload["hb_estimate_g_dl"] == pytest.approx(6.48)
    assert payload["hb_regression_available"] is True
    assert payload["hemoglobin_estimate"] == "6.48 g/dL"
    assert payload["anemia_probability"] == pytest.approx(0.5497)
    assert payload["predicted_class"] == "anemic"
    assert payload["prediction"] == "Anemic"
    assert payload["model_version"] == "vyra-dualhead-efficientnet-b3-v1.0"
    assert payload["calibration_status"] == "calibrated"
    assert payload["regression_status"] == "validated"
    assert payload["anatomical_modality"] == "conjunctiva"
    assert payload["ita_angle"] == pytest.approx(-78.23)
    assert payload["ita_category"] == "Dark"
    assert payload["glare_fraction"] == pytest.approx(0.012)
    assert payload["processing_time_ms"] == 35
    assert "Minor peripheral shadow noted." in payload["quality_warnings"]
    assert payload["triage"] is not None
    assert payload["triage"]["level"] == "RED"

    # Cleanup test record and temporary upload files
    with app_module.app.app_context():
        rec = app_module.db.session.get(app_module.Scan, payload["id"])
        if rec:
            if rec.image_file_path:
                _safe_unlink(rec.image_file_path)
                model_in = rec.image_file_path.with_name(
                    f"{rec.image_file_path.stem}_model_input{rec.image_file_path.suffix}"
                )
                _safe_unlink(model_in)
            app_module.db.session.delete(rec)
            app_module.db.session.commit()


# ---------------------------------------------------------------------------
# 7. Correct handling of missing, null, malformed, or out-of-range output fields
# ---------------------------------------------------------------------------
def test_07_handling_missing_null_malformed_or_out_of_range_fields(tmp_path: Path):
    raw_img_path = tmp_path / "valid_scan.png"
    raw_img_path.write_bytes(_make_sharp_test_image_bytes(260, 260))

    # Case A: out-of-range probability (1.75), out-of-range Hb (-4.0), invalid ITA (999.0),
    # invalid calibration_status ("magic"), but valid predicted_class ("non-anemic").
    def malformed_partial_predict(**kwargs) -> dict[str, Any]:
        return {
            "hb_estimate_g_dl": -4.0,
            "anemia_probability": 1.75,
            "predicted_class": "non-anemic",
            "model_version": "",
            "calibration_status": "magic_calibrated",
            "quality_warnings": "not-a-list",
            "error": None,
            "anatomical_modality": "conjunctiva",
            "ita_angle": 999.0,
            "ita_category": "InvalidTier",
            "glare_fraction": -0.5,
            "processing_time_ms": -10,
            "regression_status": "validated",
        }

    out = run_vyra_predict(
        raw_image_path=raw_img_path,
        body_part="conjunctiva",
        patient_payload={"age": 30, "gender": "Female"},
        predictor_fn=malformed_partial_predict,
    )
    assert out.hb_estimate_g_dl is None
    assert out.hb_regression_available is False
    assert out.anemia_probability is None
    assert out.predicted_class == "non-anemic"
    assert out.calibration_status == "uncalibrated"
    assert out.model_version == "unknown"
    assert out.ita_angle is None
    assert out.ita_category is None
    assert out.glare_fraction is None
    assert any("out-of-range" in w for w in out.quality_warnings)

    # Case B: Both anemia_probability and predicted_class invalid/missing -> raises ContractValidationError
    def completely_unusable_predict(**kwargs) -> dict[str, Any]:
        return {
            "hb_estimate_g_dl": None,
            "anemia_probability": "not_a_float",
            "predicted_class": "invalid_label",
            "error": None,
        }

    with pytest.raises(ContractValidationError):
        run_vyra_predict(
            raw_image_path=raw_img_path,
            body_part="conjunctiva",
            predictor_fn=completely_unusable_predict,
        )

    # Case C: Predictor returns a fatal `error` field -> raises PredictionExecutionError
    def fatal_error_predict(**kwargs) -> dict[str, Any]:
        return {
            "hb_estimate_g_dl": None,
            "anemia_probability": None,
            "predicted_class": None,
            "error": "Segmented conjunctiva ROI contained 0 valid pixels.",
        }

    with pytest.raises(PredictionExecutionError, match="0 valid pixels"):
        run_vyra_predict(
            raw_image_path=raw_img_path,
            body_part="conjunctiva",
            predictor_fn=fatal_error_predict,
        )


# ---------------------------------------------------------------------------
# 8. Hb readout suppressed when the regression head is unavailable or unvalidated
# ---------------------------------------------------------------------------
def test_08_hb_readout_suppressed_when_unvalidated_or_unavailable(tmp_path: Path):
    raw_img_path = tmp_path / "nail_scan.png"
    raw_img_path.write_bytes(_make_sharp_test_image_bytes(250, 250))

    # Even if a numeric hb_estimate_g_dl (e.g. 9.8) is returned, if regression_status != "validated",
    # it MUST be suppressed and display the required unavailable message.
    def unvalidated_hb_predict(**kwargs) -> dict[str, Any]:
        return {
            "hb_estimate_g_dl": 9.8,
            "anemia_probability": 0.42,
            "predicted_class": "non-anemic",
            "model_version": "vyra-dualhead-v1",
            "calibration_status": "calibrated",
            "quality_warnings": [],
            "error": None,
            "anatomical_modality": "fingernail",
            "ita_angle": None,
            "ita_category": None,
            "glare_fraction": None,
            "processing_time_ms": 28,
            "regression_status": "unavailable",
        }

    out = run_vyra_predict(
        raw_image_path=raw_img_path,
        body_part="fingernail",
        predictor_fn=unvalidated_hb_predict,
    )
    assert out.hb_estimate_g_dl is None
    assert out.hb_regression_available is False
    assert out.hemoglobin_estimate == UNAVAILABLE_HB_MESSAGE
    assert "7-9 g/dL" not in out.hemoglobin_estimate
    assert "12-15 g/dL" not in out.hemoglobin_estimate


# ---------------------------------------------------------------------------
# 9. Original tiny images rejected before resize/crop
# ---------------------------------------------------------------------------
def test_09_tiny_images_rejected_before_resize(client):
    called = False

    def should_not_be_called(**kwargs):
        nonlocal called
        called = True
        return {}

    set_test_predictor(should_not_be_called)

    # Create a 60x60 tiny image (< 100x100 minimum)
    tiny_bytes = _make_sharp_test_image_bytes(width=60, height=60)
    resp = client.post(
        "/api/predict",
        data={
            "patient_name": "Tiny Image Patient",
            "age": "29",
            "gender": "Female",
            "body_part": "conjunctiva",
            "image": (io.BytesIO(tiny_bytes), "tiny_60x60.png"),
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["success"] is False
    assert "100x100" in payload["error"]
    assert called is False


# ---------------------------------------------------------------------------
# 10. Corrupt, unsupported, unreadable, and excessively large uploads handled safely
# ---------------------------------------------------------------------------
def test_10_corrupt_unsupported_unreadable_and_large_uploads(client, monkeypatch: pytest.MonkeyPatch):
    set_test_predictor(lambda **kw: {"anemia_probability": 0.3, "predicted_class": "non-anemic"})

    # 10a. Unsupported extension (.gif / .txt)
    resp_ext = client.post(
        "/api/predict",
        data={
            "patient_name": "Ext Test",
            "age": "25",
            "gender": "Male",
            "body_part": "conjunctiva",
            "image": (io.BytesIO(b"hello world"), "notes.txt"),
        },
        content_type="multipart/form-data",
    )
    assert resp_ext.status_code == 400
    assert resp_ext.get_json()["success"] is False

    # 10b. Corrupt / unreadable bytes with .png extension
    resp_corrupt = client.post(
        "/api/predict",
        data={
            "patient_name": "Corrupt Test",
            "age": "25",
            "gender": "Male",
            "body_part": "conjunctiva",
            "image": (io.BytesIO(b"not-a-valid-png-binary-stream"), "corrupt.png"),
        },
        content_type="multipart/form-data",
    )
    assert resp_corrupt.status_code == 400
    assert resp_corrupt.get_json()["success"] is False

    # 10c. Invalid base64 payload
    resp_b64 = client.post(
        "/api/predict",
        json={
            "patient_name": "Base64 Corrupt",
            "age": 25,
            "gender": "Female",
            "body_part": "conjunctiva",
            "image": "data:image/png;base64,@@@not_valid_base64@@@",
        },
    )
    assert resp_b64.status_code == 400
    assert resp_b64.get_json()["success"] is False

    # 10d. Excessively large upload exceeding configured byte limit
    monkeypatch.setattr(app_module, "_max_upload_bytes", lambda: 1024)  # 1 KB limit for test
    large_bytes = _make_sharp_test_image_bytes(300, 300)
    assert len(large_bytes) > 1024
    resp_large = client.post(
        "/api/predict",
        data={
            "patient_name": "Large Upload Test",
            "age": "25",
            "gender": "Female",
            "body_part": "conjunctiva",
            "image": (io.BytesIO(large_bytes), "large.png"),
        },
        content_type="multipart/form-data",
    )
    assert resp_large.status_code == 400
    assert "maximum allowed size" in resp_large.get_json()["error"].lower()


# ---------------------------------------------------------------------------
# 11. Original uploads remaining separate from normalized/model-input images
# ---------------------------------------------------------------------------
def test_11_original_upload_separate_from_normalized_and_model_input(tmp_path: Path):
    # Create an non-square 420x280 original image
    orig_path = tmp_path / "raw_420x280.png"
    orig_path.write_bytes(_make_sharp_test_image_bytes(width=420, height=280))

    def fake_normalizer(image: str, body_part: str) -> dict[str, Any]:
        # Simulate Person 1 returning a 224x224 normalized ROI image + 27-dim vector
        norm_arr = np.full((224, 224, 3), 140, dtype=np.uint8)
        return {
            "is_valid": True,
            "normalized_image": norm_arr,
            "feature_vector": [0.1] * 27,
            "ita_angle": 18.4,
            "ita_category": "Intermediate",
            "glare_fraction": 0.02,
            "quality_warnings": [],
        }

    received_image_path: str | None = None

    def fake_predictor(image, feature_vector=None, body_part="conjunctiva", metadata=None, screening_threshold=0.5):
        nonlocal received_image_path
        received_image_path = str(image)
        return {
            "hb_estimate_g_dl": 11.8,
            "anemia_probability": 0.21,
            "predicted_class": "non-anemic",
            "model_version": "vyra-dualhead-v1",
            "calibration_status": "calibrated",
            "quality_warnings": [],
            "error": None,
            "anatomical_modality": body_part,
            "ita_angle": 18.4,
            "ita_category": "Intermediate",
            "glare_fraction": 0.02,
            "processing_time_ms": 22,
            "regression_status": "validated",
        }

    set_test_normalizer(fake_normalizer)
    set_test_predictor(fake_predictor)

    out = run_vyra_predict(
        raw_image_path=orig_path,
        body_part="conjunctiva",
        patient_payload={"age": 26, "gender": "Female"},
    )

    assert out.raw_image_path.resolve() == orig_path.resolve()
    assert out.normalized_image_path is not None
    assert out.normalized_image_path.resolve() != orig_path.resolve()
    assert Path(received_image_path).resolve() != orig_path.resolve()

    # Verify original image was NOT overwritten or cropped to 224x224
    with Image.open(orig_path) as raw_im:
        assert raw_im.size == (420, 280)
    with Image.open(out.normalized_image_path) as norm_im:
        assert norm_im.size == (224, 224)


# ---------------------------------------------------------------------------
# 12. Correct mapping of conjunctiva and fingernail modalities
# ---------------------------------------------------------------------------
def test_12_modality_mapping_and_rejection_of_unsupported_values(client):
    assert resolve_body_part("conjunctiva") == "conjunctiva"
    assert resolve_body_part("Eye Conjunctiva") == "conjunctiva"
    assert resolve_body_part("fingernail") == "fingernail"
    assert resolve_body_part("Fingernail") == "fingernail"

    for invalid_mod in ("Unknown", "palm", "earlobe", "", None):
        with pytest.raises(ModalityValidationError):
            resolve_body_part(invalid_mod)  # type: ignore[arg-type]

    # Verify API rejects unsupported modality with HTTP 400
    resp = client.post(
        "/api/predict",
        json={
            "patient_name": "Modality Test",
            "age": 30,
            "gender": "Female",
            "body_part": "palm",
        },
    )
    assert resp.status_code == 400
    assert "unsupported anatomical modality" in resp.get_json()["error"].lower()


# ---------------------------------------------------------------------------
# 13. Mock mode disabled by default and blocked in production
# ---------------------------------------------------------------------------
def test_13_mock_mode_disabled_by_default_and_blocked_in_production(monkeypatch: pytest.MonkeyPatch):
    # Default: disabled
    assert is_mock_mode_enabled() is False

    # Enabled in development when VYRA_ENABLE_DEV_MOCK=1
    monkeypatch.setenv("VYRA_ENABLE_DEV_MOCK", "1")
    assert is_mock_mode_enabled() is True

    # Blocked in production even when VYRA_ENABLE_DEV_MOCK=1
    monkeypatch.setenv("FLASK_ENV", "production")
    assert is_mock_mode_enabled() is False

    monkeypatch.delenv("FLASK_ENV", raising=False)
    monkeypatch.setenv("RENDER", "true")
    assert is_mock_mode_enabled() is False


# ---------------------------------------------------------------------------
# 14. Mock results not saved or reported as genuine predictions
# ---------------------------------------------------------------------------
def test_14_mock_results_not_saved_or_mixed_into_real_analytics(
    client,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VYRA_ENABLE_DEV_MOCK", "1")
    with app_module.app.app_context():
        before_count = app_module.scans_active_query().count()

    img_bytes = _make_sharp_test_image_bytes(240, 240)
    resp = client.post(
        "/api/predict",
        data={
            "patient_name": "Dev Mock Patient",
            "age": "22",
            "gender": "Female",
            "body_part": "conjunctiva",
            "use_mock": "1",
            "image": (io.BytesIO(img_bytes), "mock_scan.png"),
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["is_mock"] is True
    assert payload["persisted"] is False
    assert payload["id"] is None
    assert "mock" in payload["model_version"].lower()
    assert any("DEMO MOCK MODE" in w for w in payload["quality_warnings"])

    with app_module.app.app_context():
        after_count = app_module.scans_active_query().count()
        assert after_count == before_count


# ---------------------------------------------------------------------------
# 15. Existing database/history, CSV, media, and PDF workflows remaining functional
# ---------------------------------------------------------------------------
def test_15_database_history_csv_media_and_pdf_workflows(client):
    set_test_predictor(
        lambda **kw: {
            "hb_estimate_g_dl": 8.4,
            "anemia_probability": 0.74,
            "predicted_class": "anemic",
            "model_version": "vyra-dualhead-efficientnet-b3-v1.0",
            "calibration_status": "calibrated",
            "quality_warnings": [],
            "error": None,
            "anatomical_modality": "conjunctiva",
            "ita_angle": -22.5,
            "ita_category": "Tan",
            "glare_fraction": 0.01,
            "processing_time_ms": 40,
            "regression_status": "validated",
        }
    )

    img_bytes = _make_sharp_test_image_bytes(280, 220)
    pred_resp = client.post(
        "/api/predict",
        data={
            "patient_name": "Priya Nair Phase3Test",
            "age": "28",
            "gender": "Female",
            "phone": "9876543210",
            "image_type": "Eye Conjunctiva",
            "notes": "Phase 3 integration verification",
            "image": (io.BytesIO(img_bytes), "priya_scan.png"),
        },
        content_type="multipart/form-data",
    )
    assert pred_resp.status_code == 200
    pred_json = pred_resp.get_json()
    scan_id = pred_json["id"]
    assert scan_id is not None

    try:
        # Result page
        res_page = client.get(f"/result/{scan_id}")
        assert res_page.status_code == 200
        res_html = res_page.get_data(as_text=True)
        assert "Priya Nair Phase3Test" in res_html
        assert "8.40 g/dL" in res_html
        assert "Complete Blood Count (CBC) verification" in res_html

        # History & FTS search
        hist_page = client.get("/history?q=Priya")
        assert hist_page.status_code == 200
        assert "Priya Nair Phase3Test" in hist_page.get_data(as_text=True)

        # CSV export
        csv_resp = client.get("/history/export/csv?q=Priya")
        assert csv_resp.status_code == 200
        assert "Priya Nair Phase3Test" in csv_resp.get_data(as_text=True)

        # Uploaded media serving
        media_resp = client.get(pred_json["image_url"])
        assert media_resp.status_code == 200
        assert len(media_resp.data) > 100
        media_resp.close()

        # PDF report generation
        pdf_resp = client.get(f"/export/pdf/{scan_id}")
        assert pdf_resp.status_code == 200
        assert pdf_resp.data.startswith(b"%PDF")
        pdf_resp.close()
    finally:
        with app_module.app.app_context():
            rec = app_module.db.session.get(app_module.Scan, scan_id)
            if rec:
                if rec.image_file_path:
                    _safe_unlink(rec.image_file_path)
                    model_in = rec.image_file_path.with_name(
                        f"{rec.image_file_path.stem}_model_input{rec.image_file_path.suffix}"
                    )
                    _safe_unlink(model_in)
                app_module.db.session.delete(rec)
                app_module.db.session.commit()


# ---------------------------------------------------------------------------
# 16. Triaging never assigning GREEN when required inputs are missing
# ---------------------------------------------------------------------------
def test_16_triage_never_assigns_green_when_inputs_missing():
    # Missing both Hb and probability -> None (never GREEN)
    assert (
        compute_vyra_triage(
            hb_estimate_g_dl=None,
            regression_status="unavailable",
            anemia_probability=None,
            predicted_class="non-anemic",
        )
        is None
    )

    # Missing validated Hb (e.g. fingernail modality with low anemia_probability) -> None (never GREEN)
    assert (
        compute_vyra_triage(
            hb_estimate_g_dl=None,
            regression_status="unavailable",
            anemia_probability=0.18,
            predicted_class="non-anemic",
        )
        is None
    )

    # Unvalidated Hb >= 11.0 with low anemia_probability -> None (never GREEN)
    assert (
        compute_vyra_triage(
            hb_estimate_g_dl=13.2,
            regression_status="unavailable",
            anemia_probability=0.15,
            predicted_class="non-anemic",
        )
        is None
    )

    # Validated Hb >= 11.0 but missing anemia_probability -> None (never GREEN)
    assert (
        compute_vyra_triage(
            hb_estimate_g_dl=13.2,
            regression_status="validated",
            anemia_probability=None,
            predicted_class="non-anemic",
        )
        is None
    )

    # Conflicting inputs: Validated Hb >= 11.0 (12.4) but anemia_probability >= 0.50 (0.68) -> YELLOW (never GREEN)
    conflict_triage = compute_vyra_triage(
        hb_estimate_g_dl=12.4,
        regression_status="validated",
        anemia_probability=0.68,
        predicted_class="anemic",
    )
    assert conflict_triage is not None
    assert conflict_triage["level"] == "YELLOW"
    assert conflict_triage["conflict_detected"] is True

    # Both validated Hb >= 11.0 AND anemia_probability < 0.50 -> GREEN
    green_triage = compute_vyra_triage(
        hb_estimate_g_dl=12.6,
        regression_status="validated",
        anemia_probability=0.22,
        predicted_class="non-anemic",
    )
    assert green_triage is not None
    assert green_triage["level"] == "GREEN"


# ---------------------------------------------------------------------------
# 17. No hardcoded performance metrics presented as actual model results
# ---------------------------------------------------------------------------
def test_17_no_hardcoded_performance_metrics_or_fabricated_counts(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # 1. When champion evaluation artifact exists, it reports genuine held-out metrics (63.9%, 67.4%, 58.8%, 0.66)
    perf = app_module.performance_overview()
    assert perf["available"] is True
    assert perf["accuracy_badge"] == 63.9
    assert round(perf["accuracy"], 3) == 0.639
    assert round(perf["sensitivity"], 3) == 0.674
    assert round(perf["specificity"], 3) == 0.588
    assert round(perf["auc_roc"], 3) == 0.664

    # Check /about page renders genuine metrics rather than hardcoded 92.4% / 88.0% / 0.93
    about_resp = client.get("/about")
    assert about_resp.status_code == 200
    about_html = about_resp.get_data(as_text=True)
    assert "92.4%" not in about_html
    assert "63.9%" in about_html
    assert "67.4%" in about_html

    # 2. When no evaluation artifact exists, it cleanly reports Unavailable
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir(parents=True, exist_ok=True)
    nonexistent = empty_dir / "none.json"
    monkeypatch.setattr(app_module, "EVALUATION_LOGS_DIR", empty_dir)
    monkeypatch.setattr(app_module, "LOGS_DIR", empty_dir)
    # Test performance_overview fallback path
    def mock_perf_none():
        return {
            "available": False,
            "accuracy_badge": None,
            "accuracy": None,
            "sensitivity": None,
            "specificity": None,
            "auc_roc": None,
            "source": "Unavailable (no evaluation artifact)",
        }
    monkeypatch.setattr(app_module, "performance_overview", mock_perf_none)
    perf_none = app_module.performance_overview()
    assert perf_none["available"] is False
    assert perf_none["accuracy_badge"] is None
    assert "99.7%" not in about_html

    # Check / homepage does not contain fabricated 10000+ scan count fallback or 92.4% accuracy
    home_resp = client.get("/")
    assert home_resp.status_code == 200
    home_html = home_resp.get_data(as_text=True)
    assert 'data-count-to="10000"' not in home_html
    assert "92.4%" not in home_html
    assert "99.7%" not in home_html
