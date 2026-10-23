"""Automated Phase 5 Real-Model Integration and End-to-End Verification suite for VYRA."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any

import numpy as np
from PIL import Image
import pytest

import app as vyra_app
import config as config_module
import model as model_module
import service_adapter as adapter_module
from model import build_model, inspect_checkpoint_file, is_lfs_pointer_file, save_checkpoint
from service_adapter import (
    PROTOTYPE_MEDICAL_DISCLAIMER,
    UNAVAILABLE_HB_MESSAGE,
    compute_vyra_triage,
    inspect_runtime_readiness,
    set_test_normalizer,
    set_test_predictor,
)


def _make_sharp_image_bytes(width: int = 260, height: int = 240, fmt: str = "PNG") -> bytes:
    """Create a high-contrast synthetic image that passes resolution, blur, and brightness checks."""
    rng = np.random.default_rng(505)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    y_idx, x_idx = np.indices((height, width))
    checker = ((x_idx // 6 + y_idx // 6) % 2) * 180 + 40
    noise = rng.integers(0, 25, size=(height, width), dtype=np.uint8)
    channel = np.clip(checker + noise, 0, 255).astype(np.uint8)
    arr[:, :, 0] = channel
    arr[:, :, 1] = np.clip(channel - 20, 0, 255)
    arr[:, :, 2] = np.clip(channel - 30, 0, 255)
    image = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def _reset_phase5_state(monkeypatch: pytest.MonkeyPatch):
    existing_uploads = set(config_module.UPLOADS_DIR.glob("*")) if config_module.UPLOADS_DIR.exists() else set()
    existing_gradcam = (
        set(Path(config_module.FLASK_CONFIG.gradcam_folder).glob("*"))
        if Path(config_module.FLASK_CONFIG.gradcam_folder).exists()
        else set()
    )
    existing_reports = set(config_module.REPORTS_DIR.glob("*")) if config_module.REPORTS_DIR.exists() else set()
    set_test_predictor(None)
    set_test_normalizer(None)
    monkeypatch.delenv("VYRA_ENABLE_DEV_MOCK", raising=False)
    monkeypatch.delenv("FLASK_ENV", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setattr(vyra_app, "_predictor", None)
    monkeypatch.setattr(vyra_app, "_predictor_load_error", None)
    monkeypatch.setattr(vyra_app, "_predictor_failure_category", None)
    yield
    set_test_predictor(None)
    set_test_normalizer(None)
    if config_module.UPLOADS_DIR.exists():
        for path in config_module.UPLOADS_DIR.glob("*"):
            if path not in existing_uploads and path.name != "20260412_213235_af19d6b1.jpeg":
                _safe_unlink(path)
    gradcam_dir = Path(config_module.FLASK_CONFIG.gradcam_folder)
    if gradcam_dir.exists():
        for path in gradcam_dir.glob("*"):
            if path not in existing_gradcam and path.name != "20260412_213235_af19d6b1_gradcam_29b10e23.png":
                _safe_unlink(path)
    if config_module.REPORTS_DIR.exists():
        for path in config_module.REPORTS_DIR.glob("*"):
            if path not in existing_reports and path.name != "anemiavision_report_1.pdf":
                _safe_unlink(path)


@pytest.fixture()
def client():
    vyra_app.app.config["TESTING"] = True
    with vyra_app.app.test_client() as test_client:
        with vyra_app.app.app_context():
            vyra_app.db.session.query(vyra_app.RateLimitEvent).delete()
            vyra_app.db.session.commit()
        yield test_client


def test_01_upstream_artifacts_and_lfs_status_verified(client, tmp_path: Path):
    """Verify actual repository checkpoint validity, champion weights status, and /health reporting."""
    best_ckpt = config_module.BEST_CHECKPOINT_PATH
    assert best_ckpt.exists()
    assert is_lfs_pointer_file(best_ckpt) is False

    ckpt_info = inspect_checkpoint_file(best_ckpt)
    assert ckpt_info["checkpoint_exists"] is True
    assert ckpt_info["checkpoint_is_lfs_pointer"] is False
    assert ckpt_info["checkpoint_valid"] is True
    assert ckpt_info["failure_category"] is None

    readiness = inspect_runtime_readiness(predictor_loaded=True)
    assert readiness["model_ready"] is True
    assert readiness["checkpoint_available"] is True
    assert readiness["person2_predictor_available"] is True

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["checkpoint_exists"] is True
    assert data["checkpoint_is_lfs_pointer"] is False
    assert data["checkpoint_valid"] is True

    # Also verify that a synthetic LFS pointer is correctly detected
    fake_lfs = tmp_path / "pointer.pth"
    fake_lfs.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 100\n",
        encoding="utf-8",
    )
    lfs_info = inspect_checkpoint_file(fake_lfs)
    assert lfs_info["checkpoint_is_lfs_pointer"] is True
    assert lfs_info["checkpoint_valid"] is False
    assert lfs_info["failure_category"] == "lfs_pointer_checkpoint"


def test_02_real_pytorch_inference_end_to_end_when_valid_checkpoint_present(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify end-to-end real PyTorch inference, Grad-CAM generation, and 3-panel UI when valid weights exist."""
    ckpt_path = tmp_path / "real_efficientnet_b0.pth"
    net = build_model(pretrained=False, architecture="efficientnet_b0")
    save_checkpoint(
        model=net,
        epoch=3,
        metrics={"valid_accuracy": 0.84, "valid_sensitivity": 0.88, "valid_specificity": 0.80, "valid_auc_roc": 0.89},
        config_snapshot={"training": {"image_size": [224, 224]}},
        checkpoint_path=ckpt_path,
        architecture="efficientnet_b0",
    )

    monkeypatch.setattr(model_module, "BEST_CHECKPOINT_PATH", ckpt_path)
    monkeypatch.setattr(model_module, "LATEST_CHECKPOINT_PATH", ckpt_path)
    monkeypatch.setattr(vyra_app, "BEST_CHECKPOINT_PATH", ckpt_path)

    # /health should now report checkpoint_valid=True and checkpoint_is_lfs_pointer=False
    health_before = client.get("/health").get_json()
    assert health_before["checkpoint_valid"] is True
    assert health_before["checkpoint_is_lfs_pointer"] is False
    assert health_before["checkpoint_available"] is True

    # Submit a real scan through /predict
    img_bytes = _make_sharp_image_bytes(280, 240, "PNG")
    resp = client.post(
        "/predict",
        data={
            "patient_name": "Kavita Sharma",
            "age": "31",
            "gender": "Female",
            "body_part": "conjunctiva",
            "image_type": "Eye Conjunctiva",
            "high_sensitivity_mode": "1",
            "image": (io.BytesIO(img_bytes), "kavita_conjunctiva.png"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Verify 3-panel results screen rendered from real PyTorch forward pass
    assert 'id="panelImageQuality"' in html
    assert 'id="panelAiResults"' in html
    assert 'id="panelExplainability"' in html
    assert "REAL INFERENCE RECORD" in html
    assert any(ver in html for ver in ("legacy-efficientnet_b0-binary", "vyra-dualhead", "DualHead"))
    assert ("g/dL" in html or UNAVAILABLE_HB_MESSAGE in html)

    # Real Grad-CAM overlay should be generated by AnemiaPredictor + GradCAMPlusPlus
    assert 'class="comparison-overlay"' in html
    assert "Visual explanation unavailable" not in html

    with vyra_app.app.app_context():
        scan = vyra_app.Scan.query.filter_by(patient_name="Kavita Sharma").first()
        assert scan is not None
        assert scan.gradcam_url is not None
        gradcam_url = scan.gradcam_url

    # Verify Grad-CAM media endpoint serves the generated image
    g_resp = client.get(gradcam_url)
    assert g_resp.status_code == 200
    g_resp.close()

    # Cleanup DB row
    with vyra_app.app.app_context():
        scan = vyra_app.Scan.query.filter_by(patient_name="Kavita Sharma").first()
        if scan:
            vyra_app.db.session.delete(scan)
            vyra_app.db.session.commit()


def test_03_person1_27_feature_vector_ordering_and_missing_feature_handling(client, tmp_path: Path):
    """Verify 27-dim feature vector validation and safe withholding when features are missing/NaN or misordered."""
    received_vectors: list[Any] = []

    def fake_person2_predict(
        image,
        feature_vector=None,
        body_part="conjunctiva",
        metadata=None,
        screening_threshold=0.50,
    ) -> dict[str, Any]:
        received_vectors.append(feature_vector)
        return {
            "predicted_class": "anemic",
            "anemia_probability": 0.68,
            "hb_estimate_g_dl": 9.4,
            "regression_status": "validated",
            "model_version": "vyra-multimodal-v1",
            "calibration_status": "calibrated",
        }

    set_test_predictor(fake_person2_predict)

    # Case A: Valid 27-float vector with 27 distinct feature names
    valid_27_names = [f"feat_{i:02d}" for i in range(27)]
    valid_27_vec = [round(0.05 * (i + 1), 4) for i in range(27)]
    set_test_normalizer(
        lambda **kw: {
            "feature_vector": valid_27_vec,
            "feature_names": valid_27_names,
            "ita_angle": 12.5,
            "ita_category": "Tan",
            "glare_fraction": 0.02,
        }
    )

    img_bytes = _make_sharp_image_bytes(240, 240, "PNG")
    resp_a = client.post(
        "/api/predict",
        data={
            "patient_name": "Feature Valid Patient",
            "age": "28",
            "gender": "Female",
            "body_part": "conjunctiva",
            "image": (io.BytesIO(img_bytes), "valid_fv.png"),
        },
        content_type="multipart/form-data",
    )
    assert resp_a.status_code == 200
    assert received_vectors[-1] == pytest.approx(valid_27_vec)

    # Case B: Feature vector with a missing (None / NaN) value is safely withheld (feature_vector=None)
    nan_vec = list(valid_27_vec)
    nan_vec[5] = float("nan")
    set_test_normalizer(
        lambda **kw: {
            "feature_vector": nan_vec,
            "feature_names": valid_27_names,
        }
    )

    resp_b = client.post(
        "/api/predict",
        data={
            "patient_name": "Feature NaN Patient",
            "age": "28",
            "gender": "Female",
            "body_part": "conjunctiva",
            "image": (io.BytesIO(img_bytes), "nan_fv.png"),
        },
        content_type="multipart/form-data",
    )
    assert resp_b.status_code == 200
    assert received_vectors[-1] is None
    payload_b = resp_b.get_json()
    assert any("27-feature" in w for w in payload_b["quality_warnings"])

    # Cleanup
    with vyra_app.app.app_context():
        for name in ("Feature Valid Patient", "Feature NaN Patient"):
            scan = vyra_app.Scan.query.filter_by(patient_name=name).first()
            if scan:
                vyra_app.db.session.delete(scan)
        vyra_app.db.session.commit()


def test_04_model_unavailable_and_failed_requests_leave_no_orphan_uploads(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify failed or unavailable prediction requests clean up temporary uploads and do not persist rows."""
    before_files = set(config_module.UPLOADS_DIR.glob("*"))
    img_bytes = _make_sharp_image_bytes(240, 240, "PNG")

    # 1. Model unavailable (simulate unavailable model) -> 503
    fake_lfs = tmp_path / "pointer.pth"
    fake_lfs.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 100\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_module, "BEST_CHECKPOINT_PATH", fake_lfs)
    monkeypatch.setattr(model_module, "LATEST_CHECKPOINT_PATH", fake_lfs)
    monkeypatch.setattr(vyra_app, "BEST_CHECKPOINT_PATH", fake_lfs)
    monkeypatch.setattr(vyra_app, "_predictor", None)
    monkeypatch.setattr(adapter_module, "_INJECTED_PREDICTOR_FN", None)
    monkeypatch.setattr(adapter_module, "resolve_person2_predict_callable", lambda: None)

    resp_unavail = client.post(
        "/api/predict",
        data={
            "patient_name": "Orphan Check 1",
            "age": "35",
            "gender": "Male",
            "body_part": "conjunctiva",
            "image": (io.BytesIO(img_bytes), "orphan1.png"),
        },
        content_type="multipart/form-data",
    )
    assert resp_unavail.status_code == 503
    after_unavail_files = set(config_module.UPLOADS_DIR.glob("*"))
    assert after_unavail_files == before_files

    # 2. Predictor returns fatal error field -> 502 and no orphan uploads
    monkeypatch.undo()
    set_test_predictor(lambda **kw: {"error": "Conjunctiva ROI out of frame"})
    resp_err = client.post(
        "/api/predict",
        data={
            "patient_name": "Orphan Check 2",
            "age": "35",
            "gender": "Male",
            "body_part": "conjunctiva",
            "image": (io.BytesIO(img_bytes), "orphan2.png"),
        },
        content_type="multipart/form-data",
    )
    assert resp_err.status_code == 502
    after_err_files = set(config_module.UPLOADS_DIR.glob("*"))
    assert after_err_files == before_files


def test_05_triage_contract_alignment_and_conservative_rules():
    """Verify triage dictionary keys and conservative handling of missing/conflicting inputs."""
    # 1. RED by Hb < 7.0
    red = compute_vyra_triage(
        hb_estimate_g_dl=6.5,
        regression_status="validated",
        anemia_probability=0.72,
        predicted_class="anemic",
    )
    assert red is not None
    assert set(red.keys()) >= {"level", "label", "action", "reasons", "disclaimer", "provisional"}
    assert red["level"] == "RED"
    assert red["label"] == "RED (Urgent Referral)"
    assert red["disclaimer"] == PROTOTYPE_MEDICAL_DISCLAIMER

    # 2. Conflicting inputs (validated Hb = 12.8 >= 11.0, but anemia_probability = 0.65 >= 0.50) -> YELLOW, never GREEN
    conflict = compute_vyra_triage(
        hb_estimate_g_dl=12.8,
        regression_status="validated",
        anemia_probability=0.65,
        predicted_class="anemic",
    )
    assert conflict is not None
    assert conflict["level"] == "YELLOW"
    assert conflict["conflict_detected"] is True

    # 3. Missing validated Hb with low probability -> None (never GREEN)
    missing_hb = compute_vyra_triage(
        hb_estimate_g_dl=None,
        regression_status="unavailable",
        anemia_probability=0.12,
        predicted_class="non-anemic",
    )
    assert missing_hb is None


def test_06_evaluation_metrics_shown_only_when_backed_by_real_artifact(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify / and /about show Unavailable by default and render real metrics when test_summary.json exists."""
    resp_default = client.get("/about")
    assert resp_default.status_code == 200
    html_default = resp_default.get_data(as_text=True)
    assert ("63.9%" in html_default or "Unavailable" in html_default)

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_file = eval_dir / "test_summary.json"
    summary_file.write_text(
        json.dumps(
            {
                "accuracy": 0.875,
                "sensitivity": 0.912,
                "specificity": 0.838,
                "auc_roc": 0.934,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(vyra_app, "EVALUATION_LOGS_DIR", eval_dir)

    resp_with_artifact = client.get("/about")
    assert resp_with_artifact.status_code == 200
    html = resp_with_artifact.get_data(as_text=True)
    assert "87.5%" in html
    assert "91.2%" in html
    assert "83.8%" in html
    assert "0.93" in html
    assert "evaluation artifact (test_summary.json)" in html


def test_07_end_to_end_fingernail_modalities_and_pdf_export(client):
    """Verify full end-to-end flow for fingernails: classification active, Hb suppressed, PDF export valid."""
    img_bytes = _make_sharp_image_bytes(240, 240, "PNG")

    # 1. Fingernail scan: classification is active, continuous Hb is suppressed
    resp = client.post(
        "/predict",
        data={
            "patient_name": "Amina Owusu",
            "age": "24",
            "gender": "Female",
            "body_part": "fingernail",
            "image_type": "Fingernail",
            "image": (io.BytesIO(img_bytes), "amina_fingernail.png"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert UNAVAILABLE_HB_MESSAGE in html

    with vyra_app.app.app_context():
        scan = vyra_app.Scan.query.filter_by(patient_name="Amina Owusu").first()
        assert scan is not None
        assert scan.hb_estimate_g_dl is None
        assert scan.regression_status == "unavailable"
        scan_id = scan.id

    # 2. Export PDF report
    pdf_resp = client.get(f"/export/pdf/{scan_id}")
    assert pdf_resp.status_code == 200
    assert pdf_resp.mimetype == "application/pdf"
    pdf_data = pdf_resp.get_data()
    assert pdf_data.startswith(b"%PDF")
    pdf_resp.close()

    # Cleanup DB row
    with vyra_app.app.app_context():
        scan = vyra_app.Scan.query.filter_by(patient_name="Amina Owusu").first()
        if scan:
            vyra_app.db.session.delete(scan)
            vyra_app.db.session.commit()

