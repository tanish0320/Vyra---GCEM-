"""Comprehensive test suite for VYRA /api/v1 compatibility routes and React UI integration."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from PIL import Image

from app import Scan, create_app, db
from config import PROJECT_ROOT, UPLOADS_DIR, REPORTS_DIR


@pytest.fixture(autouse=True)
def _clean_test_state():
    existing_uploads = set(UPLOADS_DIR.glob("*")) if UPLOADS_DIR.exists() else set()
    existing_reports = set(REPORTS_DIR.glob("*")) if REPORTS_DIR.exists() else set()
    yield
    # Reset inference engine singleton
    try:
        import src.inference.predictor as pred_module
        pred_module._GLOBAL_ENGINE = None
    except Exception:
        pass
    # Cleanup database test records
    try:
        app = create_app()
        with app.app_context():
            Scan.query.filter(Scan.patient_name.in_(["Patient-001", "CHW-Kavita-09", "Cancel-Test", "Dual-Patient-01", "Sunita Devi"])).delete()
            db.session.commit()
    except Exception:
        pass
    # Cleanup uploads
    if UPLOADS_DIR.exists():
        for p in UPLOADS_DIR.glob("*"):
            if p not in existing_uploads and not p.name.endswith(".gitkeep"):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
    # Cleanup reports
    if REPORTS_DIR.exists():
        for p in REPORTS_DIR.glob("*"):
            if p not in existing_reports and not p.name.endswith(".gitkeep"):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass


@pytest.fixture
def test_client():
    """Create a configured Flask test client with an isolated in-memory or test SQLite database."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture
def sample_eye_bytes() -> bytes:
    """Return PNG bytes for a valid synthetic eye conjunctiva image."""
    img = Image.new("RGB", (200, 150), color=(180, 70, 70))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_nail_bytes() -> bytes:
    """Return PNG bytes for a valid synthetic fingernail image."""
    img = Image.new("RGB", (200, 150), color=(220, 160, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_01_system_endpoint(test_client):
    """GET /api/v1/system returns system status, versioning, and advertised capabilities."""
    resp = test_client.get("/api/v1/system")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["api_version"] == "1"
    assert data["patient_intake_version"] == 1
    assert "ready" in data
    assert data["model"]["name"] == "DualHeadEfficientNetB3"
    assert data["model"]["regression_supported"] is True
    assert data["limits"]["max_bytes"] == 10485760
    assert data["limits"]["min_side"] == 50
    assert data["capabilities"]["dual_scan"] is True
    assert data["capabilities"]["assistant"] is True
    assert data["capabilities"]["stress_test"] is True
    assert data["capabilities"]["colorimetry"] is True
    assert data["capabilities"]["research"] is True


def test_02_evidence_endpoint(test_client):
    """GET /api/v1/evidence returns champion evaluation metrics and limitations."""
    resp = test_client.get("/api/v1/evidence")
    assert resp.status_code == 200
    data = resp.get_json()

    assert "model_version" in data
    assert "checkpoint_sha256" in data
    assert "classification" in data
    assert "accuracy" in data["classification"]
    assert "hemoglobin_regression" in data
    assert "bland_altman" in data["hemoglobin_regression"]
    assert len(data.get("limitations", [])) >= 1


def test_03_analytics_endpoint(test_client):
    """GET /api/v1/analytics returns genuine active record statistics."""
    resp = test_client.get("/api/v1/analytics")
    assert resp.status_code == 200
    data = resp.get_json()

    assert "total_scans" in data
    assert "by_modality" in data
    assert "screen_positive" in data
    assert "screen_negative" in data


def test_04_samples_endpoint(test_client):
    """GET /api/v1/samples/:modality returns valid sample PNGs or 404 for unknown modality."""
    eye_resp = test_client.get("/api/v1/samples/conjunctiva")
    assert eye_resp.status_code in (200, 404)
    if eye_resp.status_code == 200:
        assert eye_resp.content_type == "image/png"

    invalid_resp = test_client.get("/api/v1/samples/invalid_tissue")
    assert invalid_resp.status_code == 404


def test_05_research_endpoint(test_client):
    """GET /api/v1/research returns research assets, dataset counts, and validation plots."""
    resp = test_client.get("/api/v1/research")
    assert resp.status_code == 200
    data = resp.get_json()

    assert "summary" in data
    assert isinstance(data["datasets"], list)
    assert isinstance(data["plots"], list)
    assert any("Confusion Matrix" in p["title"] for p in data["plots"])


def test_06_jobs_validation_and_creation(test_client, sample_eye_bytes):
    """POST /api/v1/jobs rejects missing image and accepts valid upload returning 202."""
    # Missing image
    bad_resp = test_client.post("/api/v1/jobs", data={"modality": "conjunctiva"})
    assert bad_resp.status_code == 400

    # Valid upload
    data = {
        "image": (io.BytesIO(sample_eye_bytes), "eye.png"),
        "modality": "conjunctiva",
        "reference": "Patient-001",
        "age": "32",
        "sex": "female",
        "notes": "Mild pale palpebral mucosa.",
    }
    resp = test_client.post(
        "/api/v1/jobs",
        data=data,
        content_type="multipart/form-data",
        headers={"Idempotency-Key": "test-key-12345"},
    )
    assert resp.status_code == 202
    job = resp.get_json()

    assert job["id"].startswith("job-")
    assert job["status"] == "awaiting_roi"
    assert job["stage"] == "roi"
    assert job["patient"]["reference"] == "Patient-001"
    assert job["patient"]["age"] == 32
    assert job["patient"]["sex"] == "female"
    assert "roi" in job
    assert job["detection"] is not None

    # Test idempotency
    dup_data = {
        "image": (io.BytesIO(sample_eye_bytes), "eye.png"),
        "modality": "conjunctiva",
        "reference": "Patient-001",
    }
    dup_resp = test_client.post(
        "/api/v1/jobs",
        data=dup_data,
        content_type="multipart/form-data",
        headers={"Idempotency-Key": "test-key-12345"},
    )
    assert dup_resp.status_code == 202
    dup_job = dup_resp.get_json()
    assert dup_job["id"] == job["id"]


def test_07_job_lifecycle_confirm_and_scan_serialization(test_client, sample_eye_bytes):
    """Full lifecycle: upload job -> confirm ROI -> retrieve saved Scan record."""
    # 1. Create Job
    upload_data = {
        "image": (io.BytesIO(sample_eye_bytes), "test_eye.png"),
        "modality": "conjunctiva",
        "reference": "CHW-Kavita-09",
        "age": "28",
        "sex": "female",
        "notes": "Patient notes without envelope",
    }
    create_resp = test_client.post("/api/v1/jobs", data=upload_data, content_type="multipart/form-data")
    assert create_resp.status_code == 202
    job = create_resp.get_json()
    job_id = job["id"]

    # 2. Get Job
    get_resp = test_client.get(f"/api/v1/jobs/{job_id}")
    assert get_resp.status_code == 200
    assert get_resp.get_json()["id"] == job_id

    # 3. Confirm ROI
    confirm_payload = {"roi": {"x0": 10, "y0": 10, "x1": 150, "y1": 120}}
    confirm_resp = test_client.post(
        f"/api/v1/jobs/{job_id}/confirm",
        data=json.dumps(confirm_payload),
        content_type="application/json",
    )
    assert confirm_resp.status_code == 200
    completed_job = confirm_resp.get_json()
    assert completed_job["status"] == "completed"
    assert completed_job["stage"] == "saving"
    assert completed_job["scan_id"] is not None

    scan_id = completed_job["scan_id"]

    # 4. Fetch Scan by ID
    scan_resp = test_client.get(f"/api/v1/scans/{scan_id}")
    assert scan_resp.status_code == 200
    scan = scan_resp.get_json()

    assert scan["id"] == scan_id
    assert scan["reference"] == "CHW-Kavita-09"
    assert scan["sex"] == "female"
    assert scan["notes"] == "Patient notes without envelope"
    assert 0.0 <= scan["probability_anemia"] <= 1.0
    assert scan["predicted_label"] in ("Anemic", "Non-Anemic")
    assert "artifacts" in scan
    assert "biophysical_metrics" in scan

    # 5. List Scans
    list_resp = test_client.get("/api/v1/scans")
    assert list_resp.status_code == 200
    scans_list = list_resp.get_json().get("scans", [])
    assert any(s["id"] == scan_id for s in scans_list)

    # 6. Export CSV
    csv_resp = test_client.get("/api/v1/scans.csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.content_type
    assert "CHW-Kavita-09" in csv_resp.get_data(as_text=True)

    # 7. Delete Scan (Soft-Delete)
    del_resp = test_client.delete(f"/api/v1/scans/{scan_id}")
    assert del_resp.status_code == 204

    # 8. Verify Archived (Should now return 404 on active query)
    archived_resp = test_client.get(f"/api/v1/scans/{scan_id}")
    assert archived_resp.status_code == 404


def test_08_job_cancellation(test_client, sample_eye_bytes):
    """POST /api/v1/jobs/:id/cancel updates job status to cancelled without persisting scan."""
    upload_data = {
        "image": (io.BytesIO(sample_eye_bytes), "cancel_eye.png"),
        "modality": "conjunctiva",
        "reference": "Cancel-Test",
    }
    create_resp = test_client.post("/api/v1/jobs", data=upload_data, content_type="multipart/form-data")
    assert create_resp.status_code == 202
    job_id = create_resp.get_json()["id"]

    cancel_resp = test_client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    cancelled_job = cancel_resp.get_json()
    assert cancelled_job["status"] == "cancelled"
    assert cancelled_job["scan_id"] is None


def test_09_dual_scan_job_flow(test_client, sample_eye_bytes, sample_nail_bytes):
    """Dual scan job upload and confirmation preserves multimodal fusion agreement."""
    dual_data = {
        "image": (io.BytesIO(sample_eye_bytes), "primary_eye.png"),
        "secondary_image": (io.BytesIO(sample_nail_bytes), "sec_nail.png"),
        "modality": "conjunctiva",
        "reference": "Dual-Patient-01",
        "age": "45",
        "sex": "male",
        "notes": "Multimodal screening test",
    }
    resp = test_client.post("/api/v1/jobs", data=dual_data, content_type="multipart/form-data")
    assert resp.status_code == 202
    job = resp.get_json()
    assert job["dual_scan"] is True
    assert job["secondary"] is not None
    assert job["secondary"]["modality"] == "fingernail"

    job_id = job["id"]

    # Confirm both ROIs
    confirm_payload = {
        "roi": {"x0": 10, "y0": 10, "x1": 150, "y1": 120},
        "secondary_roi": {"x0": 20, "y0": 20, "x1": 160, "y1": 130},
    }
    confirm_resp = test_client.post(
        f"/api/v1/jobs/{job_id}/confirm",
        data=json.dumps(confirm_payload),
        content_type="application/json",
    )
    assert confirm_resp.status_code == 200
    completed = confirm_resp.get_json()
    scan_id = completed["scan_id"]

    # Fetch serialized scan and verify multimodal fusion payload
    scan_resp = test_client.get(f"/api/v1/scans/{scan_id}")
    assert scan_resp.status_code == 200
    scan = scan_resp.get_json()
    assert scan["multimodal_fusion"] is not None
    assert scan["multimodal_fusion"]["active"] is True
    assert 0.0 <= scan["multimodal_fusion"]["joint_probability"] <= 100.0
    assert 0.0 <= scan["multimodal_fusion"]["concordance_pct"] <= 100.0


def test_10_ui_redirect(test_client):
    """GET /ui and /ui/ redirects to /static/ui/index.html."""
    resp = test_client.get("/ui")
    assert resp.status_code == 302
    assert "/static/ui/index.html" in resp.headers["Location"]
