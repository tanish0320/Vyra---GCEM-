"""Duplicate grouping and reproducible, group-disjoint split assignment."""

from collections import Counter, defaultdict
import hashlib
import random

import cv2
import numpy as np
from scipy.fft import dctn

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = (0.7, 0.1, 0.2)


class UnionFind:
    def __init__(self, values):
        self.parent = {v: v for v in values}

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def perceptual_hashes(image_rgb: np.ndarray) -> tuple[int, ...]:
    """63 DCT bits, excluding DC, over all eight flips/90-degree rotations."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    hashes = []
    for reflected in (small, np.fliplr(small)):
        for k in range(4):
            low = dctn(np.rot90(reflected, k), norm="ortho")[:8, :8].ravel()[1:]
            bits = low > np.median(low)
            value = 0
            for bit in bits:
                value = (value << 1) | int(bit)
            hashes.append(value)
    return tuple(hashes)


class HashTree:
    """BK-tree avoids an all-pairs image matrix."""
    def __init__(self):
        self.root = None

    def add(self, value: int, record: dict):
        if self.root is None:
            self.root = [value, [record], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(record)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [record], {}]
                return
            node = node[2][distance]

    def query(self, value: int, radius: int):
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                for record in node[1]:
                    yield record, distance
            stack.extend(child for edge, child in node[2].items() if distance - radius <= edge <= distance + radius)


def audit_samples(samples: list[dict], phash_radius: int = 6) -> list[dict]:
    """Mutates records with explicit exclusion and grouping decisions."""
    ordered = sorted(samples, key=lambda r: r["sample_id"])
    exact = UnionFind(r["source_group_id"] for r in ordered)
    pixel_seen, byte_seen = {}, {}
    events = []
    for row in ordered:
        for field, seen, kind in (("source_sha256", byte_seen, "byte_exact"), ("pixel_sha256", pixel_seen, "pixel_exact")):
            digest = row.get(field)
            if not digest:
                continue
            previous = seen.get(digest)
            if previous is not None:
                exact.union(row["source_group_id"], previous["source_group_id"])
                if kind == "pixel_exact" or row["source_sha256"] == previous["source_sha256"]:
                    events.append({"kind": kind, "left_sample_id": previous["sample_id"], "right_sample_id": row["sample_id"], "distance": 0})
            else:
                seen[digest] = row
    components = defaultdict(list)
    for row in ordered:
        components[exact.find(row["source_group_id"])].append(row)
    for members in components.values():
        parts = {r["body_part"] for r in members}
        labels = {r["anemia_label"] for r in members if r.get("anemia_label") is not None}
        hb = {r["hb_g_dl"] for r in members if r.get("hb_g_dl") is not None}
        if len(parts) > 1 or len(labels) > 1 or len(hb) > 1:
            for row in members:
                row["exclusion_reason"] = "conflicting_ground_truth_in_exact_component"
                row["inclusion_status"] = "quarantined"
    split_groups = UnionFind(r["source_group_id"] for r in ordered)
    for row in ordered:
        split_groups.union(row["source_group_id"], exact.find(row["source_group_id"]))
    trees = defaultdict(HashTree)
    near_seen = set()
    verified_near_groups = set()
    for row in ordered:
        hashes = row.get("_phashes")
        if not hashes or not row.get("is_valid"):
            continue
        tree = trees[row["body_part"]]
        for previous, distance in tree.query(hashes[0], phash_radius):
            a, b = row["source_group_id"], previous["source_group_id"]
            if exact.find(a) == exact.find(b):
                continue
            pair = tuple(sorted((a, b)))
            if pair in near_seen:
                continue
            near_seen.add(pair)
            events.append({"kind": "perceptual_candidate", "left_sample_id": previous["sample_id"], "right_sample_id": row["sample_id"], "distance": distance})
            split_groups.union(a, b)
            if row["patient_id_verified"] or previous["patient_id_verified"]:
                verified_near_groups.update([exact.find(a), exact.find(b)])
        for value in set(hashes):
            tree.add(value, row)
    pixel_keeper = {}
    for row in ordered:
        if exact.find(row["source_group_id"]) in verified_near_groups and row["inclusion_status"] == "ready":
            row["inclusion_status"] = "quarantined"
            row["exclusion_reason"] = "unresolved_cross_patient_perceptual_candidate"
        if row["inclusion_status"] != "ready":
            continue
        digest = row.get("pixel_sha256")
        if digest and digest in pixel_keeper:
            row["inclusion_status"] = "duplicate"
            row["exclusion_reason"] = "exact_duplicate"
        elif digest:
            pixel_keeper[digest] = row["sample_id"]
    component_members = defaultdict(set)
    for row in ordered:
        component_members[split_groups.find(row["source_group_id"])].add(row["source_group_id"])
    group_ids = {root: "dup:" + hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:20]
                 for root, members in component_members.items()}
    for row in ordered:
        row["duplicate_group_id"] = group_ids[split_groups.find(row["source_group_id"])]
    return events


def assign_splits(samples: list[dict], seed: int = 42) -> None:
    groups = defaultdict(list)
    totals = Counter()
    for row in samples:
        row["split"] = ""
        row["split_basis"] = "verified_patient" if row["patient_id_verified"] else "unverified_source_group"
        if row["inclusion_status"] == "included":
            row["inclusion_status"] = "ready"
        if row["inclusion_status"] == "ready":
            stratum = f"{row['dataset']}|{row['cohort']}|" + (f"hb_band_{int(row['hb_g_dl'] // 2)}" if row.get("hb_g_dl") is not None else f"label_{row['anemia_label']}")
            row["_stratum"] = stratum
            totals[stratum] += 1
            totals["dataset::" + row["dataset"]] += 1
            groups[row["duplicate_group_id"]].append(row)
    randomizer = random.Random(seed)
    ordered = sorted(groups.items())
    randomizer.shuffle(ordered)
    ordered.sort(key=lambda pair: -len(pair[1]))
    assigned = [Counter() for _ in SPLIT_NAMES]
    global_counts = [0, 0, 0]
    total = sum(len(rows) for rows in groups.values())
    group_datasets = {group: {row["dataset"] for row in rows} for group, rows in groups.items()}
    remaining_by_dataset = Counter(dataset for names in group_datasets.values() for dataset in names)
    total_groups_by_dataset = remaining_by_dataset.copy()
    dataset_counts = defaultdict(lambda: [0, 0, 0])
    for index, (_, members) in enumerate(ordered):
        counts = Counter(row["_stratum"] for row in members)
        for row in members:
            counts["dataset::" + row["dataset"]] += 1
        datasets = {row["dataset"] for row in members}
        remaining = len(ordered) - index
        empty = [i for i, count in enumerate(global_counts) if count == 0]
        candidates = empty if remaining <= len(empty) else list(range(3))
        for dataset in sorted(datasets):
            missing = [i for i, count in enumerate(dataset_counts[dataset]) if count == 0]
            if total_groups_by_dataset[dataset] >= 3 and remaining_by_dataset[dataset] <= len(missing):
                candidates = [i for i in candidates if i in missing]
        if not candidates:
            raise ValueError("Group constraints cannot produce nonempty dataset partitions")
        def cost(i):
            delta = 0.0
            for stratum, count in sorted(counts.items()):
                target = totals[stratum] * SPLIT_RATIOS[i]
                before = assigned[i][stratum] - target
                weight = 1.0 if stratum.startswith("dataset::") else 0.25
                delta += weight * ((before + count) ** 2 - before ** 2) / max(target, 1.0)
            target = total * SPLIT_RATIOS[i]
            before = global_counts[i] - target
            return delta + 0.2 * ((before + len(members)) ** 2 - before ** 2) / max(target, 1.0)
        selected = min(candidates, key=lambda i: (cost(i), i))
        assigned[selected].update(counts)
        global_counts[selected] += len(members)
        for dataset in datasets:
            dataset_counts[dataset][selected] += sum(row["dataset"] == dataset for row in members)
            remaining_by_dataset[dataset] -= 1
        for row in members:
            row["split"] = SPLIT_NAMES[selected]
            row["inclusion_status"] = "included"


def leakage_check(rows: list[dict]) -> dict:
    fields = ("patient_id", "source_group_id", "duplicate_group_id")
    seen = {field: defaultdict(set) for field in fields}
    for row in rows:
        if row.get("split") not in SPLIT_NAMES:
            continue
        for field in fields:
            if field == "patient_id" and row.get("patient_id_verified") not in (True, "true"):
                continue
            value = row.get(field)
            if value:
                seen[field][value].add(row["split"])
    result = {f"{field}_overlap": sum(len(values) > 1 for values in mapping.values()) for field, mapping in seen.items()}
    result["passed"] = not any(result.values())
    result["nail_patient_independence_verified"] = False
    return result
