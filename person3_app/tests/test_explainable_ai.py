"""Unit and integration tests for Dr. VYRA Explainable AI Assistant with MediAssist Guardrails."""

import sys
from pathlib import Path

_P3_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _P3_ROOT.parent
for p in (str(_P3_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from explainable_ai import (
    EmergencyFirewall,
    DeterministicMedicalExplainer,
    explain_scan_findings,
    build_clinical_grounding_bundle,
)


class MockScan:
    id = 101
    patient_name = "Jane Doe"
    prediction = "anemic"
    confidence = 0.94
    anemic_probability = 0.82
    hb_estimate_g_dl = 9.8
    anatomical_modality = "conjunctiva"
    triage_level = "Urgent"
    biophysical_metrics = {
        "erythema_index": 28.5,
        "a_star": 9.2,
        "l_star": 64.1,
        "b_star": 18.0,
        "perfusion_status": "Severe Pallor (Microvascular Hypoperfusion)",
    }


def test_emergency_firewall_syncope():
    result = EmergencyFirewall.inspect("I fainted this morning after walking up stairs")
    assert result is not None
    assert result["is_emergency"] is True
    assert "Syncope" in result["rule"]
    assert "108" in result["hotlines"][0]


def test_emergency_firewall_hemorrhage_hindi():
    result = EmergencyFirewall.inspect("मरीज को खून की उल्टी हो रही है", language="hi")
    assert result is not None
    assert result["is_emergency"] is True
    assert "Hemorrhagic" in result["rule"]
    assert "आपातकालीन" in result["response"]


def test_emergency_firewall_clean_pass():
    result = EmergencyFirewall.inspect("Can you explain what the Erythema Index means?")
    assert result is None


def test_grounding_bundle_extraction():
    scan = MockScan()
    bundle = build_clinical_grounding_bundle(scan)
    assert bundle["modality"] == "conjunctiva"
    assert bundle["anemic_probability_pct"] == 82.0
    assert bundle["erythema_index"] == 28.5
    assert bundle["a_star"] == 9.2
    assert bundle["hb_estimate_g_dl"] == 9.8


def test_deterministic_explainer_findings():
    scan = MockScan()
    res = explain_scan_findings(scan, "What does my image specifically show?", language="en")
    assert res["is_emergency"] is False
    assert "28.5" in res["response"]  # Erythema Index grounded
    assert "82.0%" in res["response"]  # Probability grounded
    assert "Clinical Disclaimer" in res["response"]


def test_deterministic_explainer_causes():
    scan = MockScan()
    res = explain_scan_findings(scan, "Why does anemia happen?", language="en")
    assert res["is_emergency"] is False
    assert "Iron Deficiency" in res["response"]
    assert "Clinical Disclaimer" in res["response"]


def test_deterministic_explainer_hindi():
    scan = MockScan()
    res = explain_scan_findings(scan, "मुझे क्या लक्षण देखने चाहिए?", language="hi")
    assert res["is_emergency"] is False
    assert "लक्षण" in res["response"]
    assert "अस्वीकरण" in res["response"]
