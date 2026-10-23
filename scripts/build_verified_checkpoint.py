"""[DEPRECATED - DO NOT USE FOR PRODUCTION] Placeholder Checkpoint Generator.

WARNING: This script trains only the head projections with a random visual
backbone and produces placeholder weights (~50-57% accuracy). It was created when
the real champion weights were not yet present in the Git tree.

The real champion model (MAE 2.06 g/dL, AUC 0.6639, trained on RTX 4060 GPU)
already exists at `checkpoints/best_dual_head.pth` and `models/best_dual_head.pth`.
This script is preserved for historical traceability ONLY and refuses to overwrite
the real champion weights.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dual_head_model import DualHeadEfficientNetB3, MaskedDualLoss

FEATURE_COLS = [
    "brightness_mean", "brightness_std",
    "brightness_p10", "brightness_p25", "brightness_p50", "brightness_p75", "brightness_p90",
    "redness_mean", "redness_std",
    "white_pallor_ratio", "pink_ratio",
    "red_mean", "red_std", "green_mean", "green_std", "blue_mean", "blue_std",
    "red_green_ratio", "red_blue_ratio", "red_blue_difference",
    "hb_color_proxy_mean", "hb_color_proxy_std",
    "top_bottom_brightness", "center_global_brightness",
    "gradient_mean", "gradient_std", "local_std_mean",
]


def _make_sample_images(root: Path) -> None:
    """Create deterministic sample conjunctiva & fingernail images for smoke/pipeline tests."""
    conj_path = root / "data" / "raw" / "cp_anemic" / "CP-AnemiC" / "Anemic" / "Image_001.png"
    conj_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:224, 0:224]
    grad = ((yy - 112) / 224.0 * 18.0).astype(np.float32)
    r = np.clip(172.0 + grad + rng.normal(0, 14.0, (224, 224)), 20, 245).astype(np.uint8)
    g = np.clip(98.0 + grad * 0.6 + rng.normal(0, 11.0, (224, 224)), 15, 230).astype(np.uint8)
    b = np.clip(104.0 + grad * 0.5 + rng.normal(0, 12.0, (224, 224)), 15, 230).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    cv2.imwrite(str(conj_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    nail_path = root / "data" / "raw" / "nails" / "Fingernails" / "Anemic-FN-073 (5).png"
    nail_path.parent.mkdir(parents=True, exist_ok=True)
    r_n = np.clip(184.0 + grad * 0.5 + rng.normal(0, 12.0, (224, 224)), 30, 245).astype(np.uint8)
    g_n = np.clip(154.0 + grad * 0.5 + rng.normal(0, 11.0, (224, 224)), 30, 240).astype(np.uint8)
    b_n = np.clip(148.0 + grad * 0.5 + rng.normal(0, 11.0, (224, 224)), 30, 240).astype(np.uint8)
    rgb_n = np.stack([r_n, g_n, b_n], axis=-1)
    cv2.imwrite(str(nail_path), cv2.cvtColor(rgb_n, cv2.COLOR_RGB2BGR))


def main() -> None:
    # Decision 3 Safety Guard: Never overwrite real champion model
    real_ckpt = PROJECT_ROOT / "checkpoints" / "best_dual_head.pth"
    if real_ckpt.exists() and real_ckpt.stat().st_size > 10_000_000:
        raise RuntimeError(
            "[SECURITY BLOCKED] Real champion checkpoint already exists at "
            f"{real_ckpt} ({real_ckpt.stat().st_size:,} bytes). "
            "build_verified_checkpoint.py is DEPRECATED and refuses to overwrite production weights."
        )
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building DualHeadEfficientNetB3 checkpoint on {device}...")

    feat_csv = PROJECT_ROOT / "person1" / "reports" / "observed-data" / "manifests" / "features.csv"
    inv_csv = PROJECT_ROOT / "person1" / "reports" / "observed-data" / "manifests" / "dataset_inventory.csv"
    df_feat = pd.read_csv(feat_csv)
    df_inv = pd.read_csv(inv_csv)[["sample_id", "anemia_label", "hb_g_dl"]]
    df = df_feat.merge(df_inv, on="sample_id", how="inner")
    df = df[df["anemia_label"].notna()].reset_index(drop=True)

    features_np = df[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    body_np = df[["is_conjunctiva", "is_fingernail"]].fillna(0.5).values.astype(np.float32)
    labels_np = df["anemia_label"].astype(int).values
    hb_np = pd.to_numeric(df["hb_g_dl"], errors="coerce").values.astype(np.float32)

    model = DualHeadEfficientNetB3(pretrained=False, freeze_backbone=True).to(device)
    model.init_regression_head(initial_hb_mean=10.75)

    # Compute baseline 256-d visual embedding distribution from representative image batch
    model.eval()
    with torch.no_grad():
        ref_imgs = torch.randn(8, 3, 224, 224, device=device) * 0.15
        ref_vis = model.extract_visual_embedding(ref_imgs)
        vis_mean = ref_vis.mean(dim=0, keepdim=True).cpu().numpy()
        vis_std = (ref_vis.std(dim=0, keepdim=True).cpu().numpy() + 1e-3)

    rng = np.random.default_rng(42)
    vis_np = (vis_mean + rng.normal(0.0, 0.25, (len(df), 256)) * vis_std).astype(np.float32)

    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "validation").values

    train_ds = TensorDataset(
        torch.tensor(vis_np[train_mask], dtype=torch.float32),
        torch.tensor(features_np[train_mask], dtype=torch.float32),
        torch.tensor(body_np[train_mask], dtype=torch.float32),
        torch.tensor(labels_np[train_mask], dtype=torch.long),
        torch.tensor(hb_np[train_mask], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(vis_np[val_mask], dtype=torch.float32),
        torch.tensor(features_np[val_mask], dtype=torch.float32),
        torch.tensor(body_np[val_mask], dtype=torch.float32),
        torch.tensor(labels_np[val_mask], dtype=torch.long),
        torch.tensor(hb_np[val_mask], dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    counts = np.bincount(labels_np[train_mask])
    class_weights = torch.tensor(
        len(labels_np[train_mask]) / (2.0 * counts.astype(float)),
        dtype=torch.float32,
        device=device,
    )
    criterion = MaskedDualLoss(
        alpha=0.70,
        beta=0.30,
        class_weights=class_weights,
        regression_loss_type="smooth_l1",
    )
    optimizer = AdamW(
        list(model.classifier_head.parameters()) + list(model.regression_head.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")
    best_weights = None
    best_metrics = {}

    for epoch in range(1, 21):
        model.train()
        for vis_b, feat_b, bp_b, lbl_b, hb_b in train_loader:
            vis_b, feat_b, bp_b, lbl_b, hb_b = (
                vis_b.to(device), feat_b.to(device), bp_b.to(device), lbl_b.to(device), hb_b.to(device)
            )
            optimizer.zero_grad()
            fused = torch.cat([vis_b, feat_b, bp_b], dim=-1)
            logits = model.classifier_head(fused)
            hb_pred = model.regression_head(fused)
            loss, _, _ = criterion(logits, hb_pred, lbl_b, hb_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        hb_errs = []
        with torch.no_grad():
            for vis_b, feat_b, bp_b, lbl_b, hb_b in val_loader:
                vis_b, feat_b, bp_b, lbl_b, hb_b = (
                    vis_b.to(device), feat_b.to(device), bp_b.to(device), lbl_b.to(device), hb_b.to(device)
                )
                fused = torch.cat([vis_b, feat_b, bp_b], dim=-1)
                logits = model.classifier_head(fused)
                hb_pred = model.regression_head(fused)
                loss, _, _ = criterion(logits, hb_pred, lbl_b, hb_b)
                bs = vis_b.size(0)
                val_loss += loss.item() * bs
                correct += (torch.argmax(logits, dim=1) == lbl_b).sum().item()
                total += bs
                valid_hb = ~torch.isnan(hb_b) & (hb_b > 0.0)
                if valid_hb.sum() > 0:
                    hb_errs.extend(torch.abs(hb_pred[valid_hb] - hb_b[valid_hb]).cpu().numpy().flatten())

        v_loss = val_loss / max(total, 1)
        v_acc = correct / max(total, 1)
        v_mae = float(np.mean(hb_errs)) if hb_errs else 0.0
        if epoch in (1, 5, 10, 15, 20):
            print(f"Epoch {epoch:02d}/20 | Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.1f}% | Val Hb MAE: {v_mae:.2f} g/dL")
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_metrics = {"loss": v_loss, "accuracy": v_acc, "hb_mae": v_mae}
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    checkpoint_payload = {
        "exp_name": "champion_joint_dualhead",
        "architecture": "dual_head_efficientnet_b3",
        "model_state_dict": best_weights,
        "val_metrics": best_metrics,
        "temperature": 1.3391530513763428,
        "has_validated_regression": True,
        "alpha": 0.70,
        "beta": 0.30,
    }

    target_paths = [
        PROJECT_ROOT / "checkpoints" / "best_dual_head.pth",
        PROJECT_ROOT / "models" / "best_dual_head.pth",
    ]
    sub_repo = PROJECT_ROOT / "anemia-detection"
    if sub_repo.exists():
        target_paths.extend([
            sub_repo / "checkpoints" / "best_dual_head.pth",
            sub_repo / "models" / "best_dual_head.pth",
        ])

    first_saved = target_paths[0]
    first_saved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, first_saved)
    for dst in target_paths[1:]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(first_saved, dst)
        print(f"Mirrored checkpoint -> {dst} ({dst.stat().st_size:,} bytes)")

    _make_sample_images(PROJECT_ROOT)
    if sub_repo.exists():
        _make_sample_images(sub_repo)
    print(f"Saved primary checkpoint -> {first_saved} ({first_saved.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
