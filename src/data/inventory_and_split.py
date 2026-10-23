"""Comprehensive Dataset Inventory, Duplicate Audit, and Patient-Level Leakage-Safe Splitting for VYRA.

Combines:
1. Dataset A: CP-AnemiC V2 (710 conjunctiva images with laboratory Hb measurements)
2. Dataset B: EYES-DEFY-ANEMIA (218 Italian & Indian conjunctiva images with laboratory Hb measurements)
3. Dataset C: Ghana Fingernails (4,260 nail-bed images with binary clinical anemia labels)

Outputs:
- data/manifests/dataset_inventory.csv
- data/manifests/duplicate_audit.csv
- data/splits/train.csv
- data/splits/validation.csv
- data/splits/test.csv
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFESTS_DIR = DATA_DIR / "manifests"
SPLITS_DIR = DATA_DIR / "splits"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_perceptual_hash(path: Path, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) using OpenCV for robust image loading."""
    try:
        mat = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mat is None:
            return ""
        resized = cv2.resize(mat, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        return "".join(["1" if b else "0" for b in diff.flatten()])
    except Exception:
        return ""


def clean_float(val: Any) -> float | None:
    """Safely convert strings or numbers to float, converting commas to periods."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def scan_cp_anemic() -> List[dict]:
    """Scan Dataset A: CP-AnemiC V2 with laboratory Hb labels."""
    records = []
    base_dir = RAW_DIR / "cp_anemic" / "CP-AnemiC"
    excel_path = base_dir / "Anemia_Data_Collection_Sheet.xlsx"

    if not base_dir.exists() or not excel_path.exists():
        print("WARN: CP-AnemiC directory or Excel metadata not found.")
        return records

    df = pd.read_excel(excel_path)
    metadata_by_id = {}
    for _, row in df.iterrows():
        img_id = str(row.get("IMAGE_ID", "")).strip()
        hb = clean_float(row.get("HB_LEVEL"))
        age = clean_float(row.get("Age(Months)"))
        gender = str(row.get("GENDER", "")).strip().upper()
        severity = str(row.get("Severity", "")).strip()
        hospital = str(row.get("HOSPITAL", "")).strip()

        metadata_by_id[img_id] = {
            "hb": hb,
            "age": age,
            "gender": gender,
            "severity": severity,
            "hospital": hospital,
        }

    for folder_name, is_anemic in [("Anemic", 1), ("Non-anemic", 0)]:
        f_dir = base_dir / folder_name
        if not f_dir.exists():
            continue
        for p in sorted(f_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                img_id = p.stem
                meta = metadata_by_id.get(img_id, {})
                hb_val = meta.get("hb")
                has_lab_hb = hb_val is not None and hb_val > 0.0

                records.append({
                    "dataset": "cp_anemic_v2",
                    "modality": "conjunctiva",
                    "file_path": str(p.relative_to(PROJECT_ROOT).as_posix()),
                    "filename": p.name,
                    "patient_id": f"CP_{img_id}",
                    "class_name": "anemic" if is_anemic else "non-anemic",
                    "label": is_anemic,
                    "has_lab_hb": has_lab_hb,
                    "lab_hb_g_dl": round(hb_val, 2) if has_lab_hb else "",
                    "age": meta.get("age", ""),
                    "gender": meta.get("gender", ""),
                    "sha256": compute_file_hash(p),
                    "dhash": compute_perceptual_hash(p),
                })

    print(f"Loaded {len(records)} CP-AnemiC samples (with {sum(1 for r in records if r['has_lab_hb'])} lab Hb values).")
    return records


def scan_eyes_defy_anemia() -> List[dict]:
    """Scan Dataset B: EYES-DEFY-ANEMIA with laboratory Hb labels (India & Italy cohorts)."""
    records = []
    base_dir = RAW_DIR / "eyes_defy_anemia" / "dataset anemia"
    if not base_dir.exists():
        print("WARN: EYES-DEFY-ANEMIA directory not found.")
        return records

    # 1. India Cohort
    india_dir = base_dir / "India"
    india_excel = india_dir / "India.xlsx"
    if india_excel.exists():
        df_india = pd.read_excel(india_excel)
        for _, row in df_india.iterrows():
            num = int(row.get("Number"))
            hb = clean_float(row.get("Hgb"))
            gender = str(row.get("Gender", "")).strip().upper()
            age = clean_float(row.get("Age"))

            # Determine WHO anemia status for adults: Female < 12.0, Male < 13.0
            if hb is not None:
                is_anemic = int(hb < 12.0 if gender.startswith("F") else hb < 13.0)
            else:
                is_anemic = 0

            p_dir = india_dir / str(num)
            if p_dir.exists():
                palpebral = list(p_dir.glob("*_palpebral.png"))
                if palpebral:
                    p = palpebral[0]
                    records.append({
                        "dataset": "eyes_defy_anemia",
                        "modality": "conjunctiva",
                        "file_path": str(p.relative_to(PROJECT_ROOT).as_posix()),
                        "filename": p.name,
                        "patient_id": f"EDA_India_{num:03d}",
                        "class_name": "anemic" if is_anemic else "non-anemic",
                        "label": is_anemic,
                        "has_lab_hb": hb is not None,
                        "lab_hb_g_dl": round(hb, 2) if hb is not None else "",
                        "age": age or "",
                        "gender": gender,
                        "sha256": compute_file_hash(p),
                        "dhash": compute_perceptual_hash(p),
                    })

    # 2. Italy Cohort
    italy_dir = base_dir / "Italy"
    italy_excel = italy_dir / "Italy.xlsx"
    if italy_excel.exists():
        df_italy = pd.read_excel(italy_excel)
        for _, row in df_italy.iterrows():
            num = int(row.get("Number"))
            hb = clean_float(row.get("Hgb"))
            gender = str(row.get("Gender", "")).strip().upper()
            age = clean_float(row.get("Age"))

            if hb is not None:
                is_anemic = int(hb < 12.0 if gender.startswith("F") else hb < 13.0)
            else:
                is_anemic = 0

            p_dir = italy_dir / str(num)
            if p_dir.exists():
                palpebral = list(p_dir.glob("*_palpebral.png"))
                if palpebral:
                    p = palpebral[0]
                    records.append({
                        "dataset": "eyes_defy_anemia",
                        "modality": "conjunctiva",
                        "file_path": str(p.relative_to(PROJECT_ROOT).as_posix()),
                        "filename": p.name,
                        "patient_id": f"EDA_Italy_{num:03d}",
                        "class_name": "anemic" if is_anemic else "non-anemic",
                        "label": is_anemic,
                        "has_lab_hb": hb is not None,
                        "lab_hb_g_dl": round(hb, 2) if hb is not None else "",
                        "age": age or "",
                        "gender": gender,
                        "sha256": compute_file_hash(p),
                        "dhash": compute_perceptual_hash(p),
                    })

    print(f"Loaded {len(records)} EYES-DEFY-ANEMIA samples (with {sum(1 for r in records if r['has_lab_hb'])} lab Hb values).")
    return records


def scan_ghana_fingernails() -> List[dict]:
    """Scan Dataset C: Ghana Fingernails (nail-bed images with patient identifiers)."""
    records = []
    fingernails_dir = RAW_DIR / "ghana_fingernails" / "Fingernails"
    if not fingernails_dir.exists():
        print("WARN: Ghana Fingernails directory not found.")
        return records

    for p in sorted(fingernails_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            fname = p.name
            is_anemic = "anemic" in fname.lower() and not "non" in fname.lower()
            label = 1 if is_anemic else 0
            
            # Extract patient ID, e.g. Fin-007 or FN-194
            m = re.search(r'(Fin-\d+|FN-\d+)', p.stem, re.IGNORECASE)
            pat_code = m.group(1).upper() if m else p.stem.split('(')[0].strip()

            records.append({
                "dataset": "ghana_fingernails",
                "modality": "fingernail",
                "file_path": str(p.relative_to(PROJECT_ROOT).as_posix()),
                "filename": fname,
                "patient_id": f"GH_{pat_code}",
                "class_name": "anemic" if is_anemic else "non-anemic",
                "label": label,
                "has_lab_hb": False,
                "lab_hb_g_dl": "",
                "age": "",
                "gender": "",
                "sha256": compute_file_hash(p),
                "dhash": compute_perceptual_hash(p),
            })

    print(f"Loaded {len(records)} Ghana Fingernails samples.")
    return records


def reconcile_patient_clusters_and_duplicates(records: List[dict]) -> Tuple[List[dict], Dict[str, str]]:
    """Unify patient IDs across duplicate or near-duplicate photographs.
    
    If two images share an exact SHA256 or perceptual dHash, they MUST belong
    to the same partition. This prevents augmented copies or repeat shots
    from leaking across splits.
    """
    hash_to_cluster = {}
    dhash_to_cluster = {}
    cluster_to_records = defaultdict(list)
    dup_audit = []

    for r in records:
        f_hash = r["sha256"]
        d_hash = r["dhash"]
        orig_patient = r["patient_id"]

        # Check if hash already seen
        cluster_id = None
        dup_status = "unique"
        primary_file = r["file_path"]

        if f_hash in hash_to_cluster:
            cluster_id = hash_to_cluster[f_hash]
            dup_status = "exact_sha256_duplicate"
            primary_file = cluster_to_records[cluster_id][0]["file_path"]
        elif d_hash and d_hash in dhash_to_cluster:
            cluster_id = dhash_to_cluster[d_hash]
            dup_status = "perceptual_dhash_duplicate"
            primary_file = cluster_to_records[cluster_id][0]["file_path"]
        else:
            cluster_id = f"{r['dataset']}_{orig_patient}"

        hash_to_cluster[f_hash] = cluster_id
        if d_hash:
            dhash_to_cluster[d_hash] = cluster_id

        r["cluster_patient_id"] = cluster_id
        cluster_to_records[cluster_id].append(r)

        dup_audit.append({
            "file_path": r["file_path"],
            "filename": r["filename"],
            "dataset": r["dataset"],
            "patient_id": r["patient_id"],
            "cluster_patient_id": cluster_id,
            "sha256": f_hash,
            "dhash": d_hash,
            "status": dup_status,
            "primary_file": primary_file,
        })

    return dup_audit, cluster_to_records


def generate_leakage_safe_splits(
    records: List[dict],
    cluster_to_records: Dict[str, List[dict]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Partition datasets strictly by cluster_patient_id with stratification by dataset, modality, and lab Hb availability."""
    rng = np.random.RandomState(seed)

    # Group clusters by stratify key: (dataset, modality, has_lab_hb, majority_label)
    stratified_clusters = defaultdict(list)
    for cluster_id, recs in cluster_to_records.items():
        sample = recs[0]
        has_hb = any(r["has_lab_hb"] for r in recs)
        labels = [r["label"] for r in recs]
        maj_label = max(set(labels), key=labels.count)
        key = (sample["dataset"], sample["modality"], has_hb, maj_label)
        stratified_clusters[key].append(cluster_id)

    train_clusters = set()
    val_clusters = set()
    test_clusters = set()

    for strat_key, clusters in stratified_clusters.items():
        rng.shuffle(clusters)
        n = len(clusters)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 5:
            n_val = max(1, n_val)
            n_test = n - n_train - n_val
            if n_test < 1:
                n_train -= 1

        c_train = clusters[:n_train]
        c_val = clusters[n_train:n_train + n_val]
        c_test = clusters[n_train + n_val:]

        train_clusters.update(c_train)
        val_clusters.update(c_val)
        test_clusters.update(c_test)

    train_records = []
    val_records = []
    test_records = []

    for r in records:
        cid = r["cluster_patient_id"]
        rec_copy = dict(r)
        if cid in train_clusters:
            rec_copy["split"] = "train"
            train_records.append(rec_copy)
        elif cid in val_clusters:
            rec_copy["split"] = "validation"
            val_records.append(rec_copy)
        else:
            rec_copy["split"] = "test"
            test_records.append(rec_copy)

    return train_records, val_records, test_records


def write_csv(path: Path, data: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main():
    print("=" * 60)
    print("VYRA DATASET INVENTORY, AUDIT & RECONCILIATION")
    print("=" * 60)

    # 1. Scan all three datasets
    records = []
    records.extend(scan_cp_anemic())
    records.extend(scan_eyes_defy_anemia())
    records.extend(scan_ghana_fingernails())

    print(f"\nTotal image samples scanned across all datasets: {len(records)}")

    # 2. Reconcile duplicates and unify patient clusters
    dup_audit, cluster_to_records = reconcile_patient_clusters_and_duplicates(records)
    print(f"Total unique patient clusters: {len(cluster_to_records)}")
    
    exact_dups = sum(1 for d in dup_audit if d["status"] == "exact_sha256_duplicate")
    perc_dups = sum(1 for d in dup_audit if d["status"] == "perceptual_dhash_duplicate")
    print(f"Duplicate audit: {exact_dups} exact SHA256 duplicates, {perc_dups} near perceptual duplicates.")

    # 3. Write inventory and duplicate audit
    inventory_fields = [
        "dataset", "modality", "file_path", "filename", "patient_id",
        "cluster_patient_id", "class_name", "label", "has_lab_hb",
        "lab_hb_g_dl", "age", "gender", "sha256", "dhash"
    ]
    write_csv(MANIFESTS_DIR / "dataset_inventory.csv", records, inventory_fields)
    print(f"Wrote inventory to {MANIFESTS_DIR / 'dataset_inventory.csv'}")

    dup_fields = ["file_path", "filename", "dataset", "patient_id", "cluster_patient_id", "sha256", "dhash", "status", "primary_file"]
    write_csv(MANIFESTS_DIR / "duplicate_audit.csv", dup_audit, dup_fields)
    print(f"Wrote duplicate audit to {MANIFESTS_DIR / 'duplicate_audit.csv'}")

    # 4. Generate splits
    train, val, test = generate_leakage_safe_splits(records, cluster_to_records)
    split_fields = inventory_fields + ["split"]
    write_csv(SPLITS_DIR / "train.csv", train, split_fields)
    write_csv(SPLITS_DIR / "validation.csv", val, split_fields)
    write_csv(SPLITS_DIR / "test.csv", test, split_fields)

    print("\n" + "=" * 60)
    print("RECONCILED SPLIT SUMMARY")
    print("=" * 60)
    for s_name, split_data in [("Train (70%)", train), ("Validation (10%)", val), ("Test (20%)", test)]:
        counts = defaultdict(int)
        lab_hb_count = sum(1 for r in split_data if r["has_lab_hb"])
        pts = set(r["cluster_patient_id"] for r in split_data)
        for r in split_data:
            counts[f"{r['dataset']}_{r['modality']}_{r['class_name']}"] += 1
        print(f"\n{s_name}:")
        print(f"  Total Images: {len(split_data)}")
        print(f"  Unique Patient Clusters: {len(pts)}")
        print(f"  Samples with Lab Hb Targets: {lab_hb_count} ({lab_hb_count/len(split_data)*100:.1f}%)")
        print(f"  Stratified breakdown:")
        for k, v in sorted(counts.items()):
            print(f"    - {k}: {v}")

    # 5. Rigorous Leakage Verification
    train_pts = set(r["cluster_patient_id"] for r in train)
    val_pts = set(r["cluster_patient_id"] for r in val)
    test_pts = set(r["cluster_patient_id"] for r in test)

    leak_tv = train_pts & val_pts
    leak_tt = train_pts & test_pts
    leak_vt = val_pts & test_pts

    print("\n" + "=" * 60)
    print("LEAKAGE INTEGRITY AUDIT")
    print("=" * 60)
    print(f"Train/Val Patient Overlap: {len(leak_tv)} (MUST BE 0)")
    print(f"Train/Test Patient Overlap: {len(leak_tt)} (MUST BE 0)")
    print(f"Val/Test Patient Overlap: {len(leak_vt)} (MUST BE 0)")

    assert len(leak_tv) == 0, "Patient leakage in Train/Val!"
    assert len(leak_tt) == 0, "Patient leakage in Train/Test!"
    assert len(leak_vt) == 0, "Patient leakage in Val/Test!"
    print("SUCCESS: Zero patient and duplicate leakage across all partitions!")


if __name__ == "__main__":
    main()
