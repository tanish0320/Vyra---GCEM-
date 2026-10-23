from copy import deepcopy
import random

import numpy as np

from vyra_preprocessing.audit import assign_splits, audit_samples, leakage_check, perceptual_hashes


def sample(identifier, *, patient=True, label=None, hb=12.0, pixel=None):
    return {
        "sample_id": identifier, "dataset": "eyes_defy_anemia" if patient else "ghana_fingernails",
        "cohort": "india" if patient else "ghana", "body_part": "conjunctiva" if patient else "fingernail",
        "source_group_id": "source:" + identifier, "patient_id": "patient:" + identifier if patient else None,
        "patient_id_verified": patient, "hb_g_dl": hb if patient else None, "anemia_label": label,
        "source_sha256": "bytes:" + identifier, "pixel_sha256": pixel or "pixels:" + identifier,
        "inclusion_status": "ready", "exclusion_reason": "", "is_valid": True,
    }


def test_exact_duplicate_groups_never_cross_splits():
    rows = [sample(str(i)) for i in range(20)]
    rows[1]["pixel_sha256"] = rows[0]["pixel_sha256"]
    audit_samples(rows)
    assert rows[0]["duplicate_group_id"] == rows[1]["duplicate_group_id"]
    assert sum(r["inclusion_status"] == "duplicate" for r in rows) == 1
    assign_splits(rows)
    assert leakage_check(rows)["passed"]
    assert {r["split"] for r in rows if r["split"]} == {"train", "validation", "test"}


def test_conflicting_hb_on_same_image_is_quarantined():
    rows = [sample("one", hb=8, pixel="same"), sample("two", hb=15, pixel="same")]
    audit_samples(rows)
    assert all(r["inclusion_status"] == "quarantined" for r in rows)


def test_conflicting_nail_class_on_same_image_is_quarantined():
    rows = [sample("one", patient=False, label=0, pixel="same"), sample("two", patient=False, label=1, pixel="same")]
    audit_samples(rows)
    assert all(r["inclusion_status"] == "quarantined" for r in rows)


def test_near_matches_quarantined_for_verified_patients_but_grouped_for_exploration():
    eyes = [sample("one"), sample("two")]
    for row in eyes:
        row["_phashes"] = (123,)
    events = audit_samples(eyes)
    assert any(e["kind"] == "perceptual_candidate" for e in events)
    assert all(r["inclusion_status"] == "quarantined" for r in eyes)
    nails = [sample("one", patient=False, label=0), sample("two", patient=False, label=1)]
    for row in nails:
        row["_phashes"] = (123,)
    audit_samples(nails)
    assert nails[0]["duplicate_group_id"] == nails[1]["duplicate_group_id"]
    assert all(r["inclusion_status"] == "ready" for r in nails)
    assign_splits(nails)
    assert all(r["split_basis"] == "unverified_source_group" for r in nails)


def test_perceptual_hash_orbits_cover_rotations_and_reflections():
    image = np.random.default_rng(13).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    base = set(perceptual_hashes(image))
    assert base == set(perceptual_hashes(np.rot90(image)))
    assert base == set(perceptual_hashes(np.fliplr(image)))


def test_split_reproducible_even_when_scan_order_changes():
    rows = [sample(str(i), hb=8 + i / 3) for i in range(30)]
    other = deepcopy(rows)
    random.Random(22).shuffle(other)
    for records in (rows, other):
        audit_samples(records)
        assign_splits(records, 42)
    assert {r["sample_id"]: r["split"] for r in rows} == {r["sample_id"]: r["split"] for r in other}


def test_overlap_check_fails_for_known_patient_or_duplicate_crossing():
    rows = [sample("one"), sample("two")]
    for row, split in zip(rows, ["train", "test"]):
        row.update(split=split, duplicate_group_id="same", patient_id="same_patient")
    result = leakage_check(rows)
    assert not result["passed"] and result["patient_id_overlap"] == 1 and result["duplicate_group_id_overlap"] == 1

