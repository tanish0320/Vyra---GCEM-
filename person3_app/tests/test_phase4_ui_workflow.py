"""Automated Phase 4 CHW workflow UI and 3-panel results experience tests for VYRA."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest

import app as vyra_app
import config as config_module
from service_adapter import (
    UNAVAILABLE_HB_MESSAGE,
    set_test_normalizer,
    set_test_predictor,
)


def _make_sharp_image_bytes(width: int = 240, height: int = 240, fmt: str = "PNG") -> bytes:
    """Create a synthetic high-contrast checkerboard image that passes blur & brightness checks."""
    rng = np.random.default_rng(104)
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
def _reset_phase4_state(monkeypatch: pytest.MonkeyPatch):
    existing_uploads = set(config_module.UPLOADS_DIR.glob("*")) if config_module.UPLOADS_DIR.exists() else set()
    existing_reports = set(config_module.REPORTS_DIR.glob("*")) if config_module.REPORTS_DIR.exists() else set()
    set_test_predictor(None)
    set_test_normalizer(None)
    monkeypatch.setattr(vyra_app, "_predictor", None)
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
    vyra_app.app.config["TESTING"] = True
    with vyra_app.app.test_client() as test_client:
        with vyra_app.app.app_context():
            vyra_app.db.session.query(vyra_app.RateLimitEvent).delete()
            vyra_app.db.session.commit()
        yield test_client


def test_scan_page_renders_chw_intake_modality_and_honest_lfs_state(client):
    """Verify /scan renders modality cards, intake fields, 3-stage processing overlay, and honest LFS state."""
    resp = client.get("/scan")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Honest Model Unavailable state when checkpoint is Git LFS pointer
    assert "Model Unavailable" in html
    assert "Git LFS Pointer Detected" in html
    assert 'data-model-state="unavailable"' in html

    # Modality selector cards & guidance
    assert 'data-modality="conjunctiva"' in html
    assert 'data-modality="fingernail"' in html
    assert "Conjunctiva capture guidance:" in html

    # High-sensitivity CHW triage mode option
    assert 'name="high_sensitivity_mode"' in html
    assert "0.255" in html

    # Honest 3-stage processing overlay
    assert "1. Image Upload &amp; Quality Gate" in html
    assert "2. Optical Normalization &amp; Feature Extraction" in html
    assert "3. AI Screening &amp; Triage" in html


def test_scan_page_renders_model_ready_when_predictor_available(client):
    """Verify /scan shows Model Ready when Person 2 predictor callable is available."""
    set_test_predictor(lambda **kw: {})

    resp = client.get("/scan")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Model Ready" in html
    assert 'data-model-state="ready"' in html


def test_three_panel_results_screen_without_normalization_hb_or_gradcam(client):
    """Verify the 3-panel results screen and separate referral card when optional upstream outputs are absent."""

    def fake_person2_predict(
        image,
        feature_vector=None,
        body_part="conjunctiva",
        metadata=None,
        screening_threshold=0.50,
    ) -> dict[str, Any]:
        return {
            "predicted_class": "anemic",
            "anemia_probability": 0.78,
            "confidence": 0.78,
            "hb_estimate_g_dl": None,
            "regression_status": "unavailable",
            "model_version": "vyra-conjunctiva-v1",
            "calibration_status": "calibrated",
            "quality_warnings": ["Slightly soft focus near lateral canthus"],
        }

    set_test_predictor(fake_person2_predict)

    img_bytes = _make_sharp_image_bytes(240, 240, "PNG")
    resp = client.post(
        "/predict",
        data={
            "patient_name": "Sunita Devi",
            "age": "29",
            "gender": "Female",
            "phone": "9876500001",
            "body_part": "conjunctiva",
            "image_type": "Eye Conjunctiva",
            "notes": "Postpartum screening visit",
            "image": (io.BytesIO(img_bytes), "conjunctiva.png"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Three panels exist
    assert 'id="panelImageQuality"' in html
    assert 'id="panelAiResults"' in html
    assert 'id="panelExplainability"' in html

    # Panel 1: Modality, warnings, and honest absence of normalized image
    assert "Conjunctiva" in html
    assert "Slightly soft focus near lateral canthus" in html
    assert "Normalized ROI image unavailable" in html

    # Panel 2: AI results, probability, model version, calibration status, and unavailable Hb message
    assert "Anemic" in html
    assert "78.0%" in html
    assert "vyra-conjunctiva-v1" in html
    assert "calibrated" in html
    assert UNAVAILABLE_HB_MESSAGE in html

    # Panel 3: Honest explainability unavailable message (no fake heatmap)
    assert "Visual explanation unavailable" in html
    assert 'class="comparison-overlay"' not in html

    # Separate Referral Card with provisional triage & mandatory disclaimer
    assert 'id="referralWorkflowCard"' in html
    assert "YELLOW (PHC Referral)" in html
    assert "Complete Blood Count (CBC)" in html
    assert "Postpartum screening visit" in html

    # Cleanup test record
    with vyra_app.app.app_context():
        scan = vyra_app.Scan.query.filter_by(patient_name="Sunita Devi").first()
        assert scan is not None
        vyra_app.db.session.delete(scan)
        vyra_app.db.session.commit()


def test_three_panel_results_with_validated_hb_person1_normalization_and_gradcam(client, tmp_path):
    """Verify Panel 1, Panel 2, Panel 3, and GREEN triage when validated Hb and normalization exist."""

    norm_img_path = tmp_path / "normalized_roi.png"
    norm_img_path.write_bytes(_make_sharp_image_bytes(224, 224, "PNG"))

    def fake_person1_normalizer(**kwargs: Any) -> dict[str, Any]:
        return {
            "normalized_image": str(norm_img_path),
            "feature_vector": [0.15] * 27,
            "ita_angle": 18.4,
            "ita_category": "Intermediate",
            "glare_fraction": 0.032,
            "quality_warnings": [],
        }

    def fake_person2_predict(
        image,
        feature_vector=None,
        body_part="fingernail",
        metadata=None,
        screening_threshold=0.50,
    ) -> dict[str, Any]:
        return {
            "predicted_class": "non-anemic",
            "anemia_probability": 0.14,
            "confidence": 0.86,
            "hb_estimate_g_dl": 13.4,
            "regression_status": "validated",
            "model_version": "vyra-fingernail-v2",
            "calibration_status": "calibrated",
            "quality_warnings": [],
        }

    set_test_normalizer(fake_person1_normalizer)
    set_test_predictor(fake_person2_predict)

    img_bytes = _make_sharp_image_bytes(240, 240, "PNG")
    resp = client.post(
        "/predict",
        data={
            "patient_name": "Kiran Rao",
            "age": "34",
            "gender": "Male",
            "body_part": "fingernail",
            "image_type": "Fingernail",
            "image": (io.BytesIO(img_bytes), "fingernail.png"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Panel 1: Normalized ROI image & optical metrics rendered
    assert "Normalized ROI Image (Person 1 Pipeline)" in html
    assert "18.4°" in html
    assert "Intermediate" in html
    assert "3.2%" in html

    # Panel 2: Validated numeric Hb estimate rendered
    assert "13.40 g/dL" in html
    assert "Validated Regression Output" in html
    assert "vyra-fingernail-v2" in html

    # Referral Card: GREEN clearance allowed because validated Hb >= 11.0 g/dL and prob < 0.50
    assert "GREEN (Routine Follow-up)" in html

    # Cleanup
    with vyra_app.app.app_context():
        scan = vyra_app.Scan.query.filter_by(patient_name="Kiran Rao").first()
        assert scan is not None
        vyra_app.db.session.delete(scan)
        vyra_app.db.session.commit()


def test_conservative_referral_card_withholds_green_when_hb_unvalidated(client):
    """Verify a non-anemic prediction without validated Hb shows Incomplete/Clinical Review instead of GREEN."""

    def fake_person2_predict(
        image,
        feature_vector=None,
        body_part="conjunctiva",
        metadata=None,
        screening_threshold=0.50,
    ) -> dict[str, Any]:
        return {
            "predicted_class": "non-anemic",
            "anemia_probability": 0.18,
            "confidence": 0.82,
            "hb_estimate_g_dl": None,
            "regression_status": "unavailable",
            "model_version": "vyra-conjunctiva-v1",
            "calibration_status": "calibrated",
        }

    set_test_predictor(fake_person2_predict)

    img_bytes = _make_sharp_image_bytes(240, 240, "PNG")
    resp = client.post(
        "/predict",
        data={
            "patient_name": "Meera Nair",
            "age": "26",
            "gender": "Female",
            "body_part": "conjunctiva",
            "image": (io.BytesIO(img_bytes), "conjunctiva.png"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    assert "Provisional Triage: Incomplete · Clinical Review Required" in html
    assert "GREEN · Normal / Routine Monitoring" not in html

    with vyra_app.app.app_context():
        scan = vyra_app.Scan.query.filter_by(patient_name="Meera Nair").first()
        if scan:
            vyra_app.db.session.delete(scan)
            vyra_app.db.session.commit()


def test_history_and_pdf_report_preserve_genuine_records_and_disclaimer(client):
    """Verify CHW can find and open prior genuine records in /history and download PDF with disclaimer."""

    def fake_person2_predict(
        image,
        feature_vector=None,
        body_part="fingernail",
        metadata=None,
        screening_threshold=0.50,
    ) -> dict[str, Any]:
        return {
            "predicted_class": "anemic",
            "anemia_probability": 0.64,
            "confidence": 0.64,
            "hb_estimate_g_dl": None,
            "regression_status": "unavailable",
            "model_version": "vyra-conjunctiva-v1",
            "calibration_status": "calibrated",
        }

    set_test_predictor(fake_person2_predict)

    img_bytes = _make_sharp_image_bytes(240, 240, "PNG")
    create_resp = client.post(
        "/api/predict",
        data={
            "patient_name": "Lakshmi Bai",
            "age": "41",
            "gender": "Female",
            "body_part": "fingernail",
            "image": (io.BytesIO(img_bytes), "nail.png"),
        },
        content_type="multipart/form-data",
    )
    assert create_resp.status_code == 200
    scan_id = create_resp.get_json()["scan_id"]

    # Check /history search finds the genuine record and displays modality
    hist_resp = client.get("/history?q=Lakshmi")
    assert hist_resp.status_code == 200
    hist_html = hist_resp.get_data(as_text=True)
    assert "Lakshmi Bai" in hist_html
    assert "Fingernail" in hist_html
    assert f"/result/{scan_id}" in hist_html

    # Open the result screen from history
    res_resp = client.get(f"/result/{scan_id}")
    assert res_resp.status_code == 200
    res_html = res_resp.get_data(as_text=True)
    assert "Lakshmi Bai" in res_html
    assert "YELLOW (PHC Referral)" in res_html

    # Download PDF report and close response handle cleanly
    pdf_resp = client.get(f"/export/pdf/{scan_id}")
    assert pdf_resp.status_code == 200
    assert pdf_resp.mimetype == "application/pdf"
    assert pdf_resp.data.startswith(b"%PDF-")
    pdf_resp.close()

    # Cleanup
    with vyra_app.app.app_context():
        scan = vyra_app.db.session.get(vyra_app.Scan, scan_id)
        if scan:
            vyra_app.db.session.delete(scan)
            vyra_app.db.session.commit()


def test_no_patient_pii_logged_in_frontend_javascript():
    """Ensure frontend JavaScript files do not log patient-sensitive data to the browser console."""
    js_dir = Path(vyra_app.STATIC_DIR) / "js"
    for js_file in js_dir.glob("*.js"):
        source = js_file.read_text(encoding="utf-8")
        assert "console.log(" not in source, f"Found console.log in {js_file.name}"
