import csv
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import numpy as np
from openpyxl import Workbook
from PIL import Image
import pytest

from vyra_preprocessing.archives import extract_zip
from vyra_preprocessing.datasets import export_handoff, eye_role, nail_identity, parse_number, prepare_datasets, verify_prepared
from vyra_preprocessing.prepared import load_prepared_sample


def _synthetic_sources(root: Path):
    """Generated test pixels and labels, not clinical examples or model outputs."""
    eyes, nails = root / "eyes", root / "nails"
    nails.mkdir(parents=True)
    randomizer = np.random.default_rng(15)
    for country in ("India", "Italy"):
        cohort = eyes / country
        cohort.mkdir(parents=True)
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["Number", "Hgb", "Gender", "Age", "Note"])
        for number in range(1, 7):
            folder = cohort / str(number)
            folder.mkdir()
            image = np.full((80, 80, 3), 255, dtype=np.uint8)
            image[10:70, 10:70] = randomizer.integers(50, 200, (60, 60, 3), dtype=np.uint8)
            Image.fromarray(image).save(folder / f"{number}.jpg")
            suffix = "papebral" if country == "India" and number == 1 else "palpebral"
            Image.fromarray(image).save(folder / f"{number}_{suffix}.png")
            sheet.append([float(number), "_" if country == "Italy" and number == 6 else f"{9 + number},1", "F", 25, "synthetic test metadata"])
        workbook.save(cohort / f"{country}.xlsx")
    for number in range(1, 5):
        rgba = np.zeros((80, 80, 4), dtype=np.uint8)
        rgba[10:70, 10:70, :3] = randomizer.integers(50, 200, (60, 60, 3), dtype=np.uint8)
        rgba[10:70, 10:70, 3] = 255
        Image.fromarray(rgba).save(nails / f"Anemic-FN-{number:03}.png")
    Image.fromarray(np.full((40, 70, 3), (170, 95, 100), dtype=np.uint8)).save(nails / "Non-Anrmic-FN-009.png")
    shutil.copyfile(nails / "Anemic-FN-001.png", nails / "Anmeic-fn-0001 1.png")
    return eyes, nails


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("generated-datasets")
    eyes, nails = _synthetic_sources(root)
    output = root / "prepared"
    summary = prepare_datasets(eyes, nails, output, progress=lambda _: None)
    return eyes, nails, output, summary


def test_decimal_commas_and_missing_values():
    assert parse_number("15,1") == 15.1
    assert parse_number("_") is None
    assert parse_number(None) is None


@pytest.mark.parametrize("filename,expected", [
    ("Anemic-Fin-007 (12).png", ("ghana:source:1:7", 1)),
    ("Anmeic-fn-0001 1.png", ("ghana:source:1:1", 1)),
    ("Non-Anrmic-FN-204 (6).png", ("ghana:source:0:204", 0)),
    ("Anemic-FN-011FV (17).png", ("ghana:source:1:11", 1)),
    ("Anemic-FN-011RA.png", ("ghana:source:1:11", 1)),
    ("Anmeic-fn-0021 1 5.png", ("ghana:source:1:21", 1)),
])
def test_known_nail_aliases(filename, expected):
    assert nail_identity(filename) == expected


@pytest.mark.parametrize("name,role", [
    ("capture_papebral.png", "palpebral"),
    ("capture_palplebral.png", "palpebral"),
    ("capture_forniceal_palplebral.png", "forniceal_palpebral"),
    ("capture_palpebral(1).png", "palpebral"),
])
def test_observed_eye_filename_typos(name, role):
    assert eye_role(name) == role


def test_real_preparation_behavior_on_generated_sources(prepared):
    _, _, output, summary = prepared
    assert summary["inventory_image_count"] == 30
    assert summary["eye_patient_count"] == 12
    assert summary["eye_patients_with_lab_hb"] == 11
    assert summary["datasets"]["ghana_fingernails"]["status_counts"]["invalid"] == 1
    assert summary["leakage"]["passed"]
    assert not summary["leakage"]["nail_patient_independence_verified"]
    assert verify_prepared(output)["passed"]
    with (output / "manifests" / "sample_manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(not r["hb_g_dl"] and not r["patient_id"] for r in rows if r["dataset"] == "ghana_fingernails")
    assert all(r["patient_id_verified"] == "true" for split in ("train", "validation", "test")
               for r in csv.DictReader((output / "verified_splits" / f"{split}.csv").open()))


def test_cached_and_uncached_consumers_match_for_both_modalities(prepared):
    _, _, output, _ = prepared
    with (output / "manifests" / "sample_manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for part in ("conjunctiva", "fingernail"):
        row = next(r for r in rows if r["split"] and r["body_part"] == part)
        cached = load_prepared_sample(row, output)
        recomputed = load_prepared_sample(row, output, cached=False)
        assert cached.is_valid and recomputed.is_valid
        np.testing.assert_array_equal(cached.normalized_image_rgb, recomputed.normalized_image_rgb)
        np.testing.assert_array_equal(cached.feature_vector, recomputed.feature_vector)


def test_repeat_run_verifies_and_reuses(prepared):
    eyes, nails, output, summary = prepared
    assert prepare_datasets(eyes, nails, output, progress=lambda _: None) == summary


def test_reproducible_manifests_across_output_paths(prepared, tmp_path):
    eyes, nails, output, _ = prepared
    other = tmp_path / "second"
    prepare_datasets(eyes, nails, other, progress=lambda _: None)
    for name in ["sample_manifest.csv", "dataset_inventory.csv", "features.csv", "duplicate_audit.csv"]:
        assert (output / "manifests" / name).read_bytes() == (other / "manifests" / name).read_bytes()


def test_changed_source_cannot_reuse_old_output(prepared, tmp_path):
    eyes, nails, output, _ = prepared
    changed = tmp_path / "changed-nails"
    shutil.copytree(nails, changed)
    (changed / "extra.txt").write_text("changed provenance")
    with pytest.raises(ValueError, match="Output already exists"):
        prepare_datasets(eyes, changed, output, progress=lambda _: None)


def test_tampered_normalized_image_fails_verification(prepared, tmp_path):
    _, _, output, _ = prepared
    altered = tmp_path / "altered"
    shutil.copytree(output, altered)
    with (altered / "manifests" / "sample_manifest.csv").open(newline="") as stream:
        row = next(row for row in csv.DictReader(stream) if row["split"])
    (altered / row["normalized_path"]).write_bytes(b"changed")
    assert not verify_prepared(altered)["passed"]
    with pytest.raises(ValueError, match="hash mismatch"):
        load_prepared_sample(row, altered)


def test_export_contains_no_images_or_archives(prepared, tmp_path):
    _, _, output, _ = prepared
    target = tmp_path / "metadata-only"
    export_handoff(output, target)
    assert (target / "verification.json").is_file()
    assert all(p.suffix.lower() not in {".png", ".jpg", ".zip", ".rar"} for p in target.rglob("*") if p.is_file())


def test_handoff_export_cannot_overwrite_prepared_data(prepared):
    _, _, output, _ = prepared
    with pytest.raises(ValueError, match="must not contain"):
        export_handoff(output, output / "reports")


def test_bad_zip_paths_rejected_without_escape(tmp_path):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_zip(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()
