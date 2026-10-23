"""The Model's Achilles Heel: Shortcut Learning & Generalization Stress-Tester.

Investigates whether the screening model relies on meaningful anatomical tissue
information or superficial shortcuts (background skin, image borders, camera
defocus, or illuminant shifts).

This is a hypothesis-testing research tool, not automatic proof that a specific
feature caused the prediction.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from roi_detector import detect_and_crop_roi
from src.inference.predictor import predict


@dataclass
class PerturbationCondition:
    condition_id: str
    name: str
    category: str
    probability: float
    predicted_class: str
    delta_prob: float
    percent_shift: float
    thumbnail_b64: str
    description: str


@dataclass
class FailureMechanism:
    id: str
    name: str
    risk_level: str  # 'Critical' | 'Elevated' | 'Moderate' | 'Resilient'
    sensitivity_score: float  # 0 to 100
    evidence: str
    mechanism_detail: str
    clinical_implication: str


def _to_base64_thumbnail(img_bgr: np.ndarray, max_dim: int = 160) -> str:
    """Encode an image array into a compact base64 JPEG thumbnail."""
    h, w = img_bgr.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        thumb = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        thumb = img_bgr
    success, encoded = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def analyze_model_shortcuts(
    image_bgr: np.ndarray,
    modality: str = "conjunctiva",
) -> dict[str, Any]:
    """Execute a battery of controlled physical & optical perturbations to uncover shortcut learning."""
    h, w = image_bgr.shape[:2]
    canon_modality = "fingernail" if "nail" in str(modality).lower() else "conjunctiva"

    # 1. Base ROI extraction
    roi_res = detect_and_crop_roi(image_bgr, modality=canon_modality)
    if roi_res.success and roi_res.cropped_image_rgb is not None:
        roi_rgb = roi_res.cropped_image_rgb
        roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
    else:
        roi_bgr = cv2.resize(image_bgr, (256, 256))
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    pred_base = predict(roi_rgb, body_part=canon_modality)
    p_base = float(pred_base.get("anemia_probability", 0.5))
    class_base = str(pred_base.get("predicted_class", "unknown"))

    # 2. Raw uncropped full image
    raw_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred_raw = predict(raw_rgb, body_part=canon_modality)
    p_raw = float(pred_raw.get("anemia_probability", 0.5))
    class_raw = str(pred_raw.get("predicted_class", "unknown"))

    # 3. Tissue Only (Mask background to neutral gray)
    ch, cw = roi_bgr.shape[:2]
    mask_tissue = np.zeros((ch, cw), dtype=np.uint8)
    if canon_modality == "conjunctiva":
        # Target lower-mid horizontal mucosal band
        cv2.ellipse(mask_tissue, (cw // 2, int(ch * 0.52)), (int(cw * 0.40), int(ch * 0.28)), 0, 0, 360, 255, -1)
    else:
        # Target distal nail plate oval
        cv2.ellipse(mask_tissue, (cw // 2, int(ch * 0.50)), (int(cw * 0.35), int(ch * 0.38)), 0, 0, 360, 255, -1)

    tissue_only_bgr = cv2.bitwise_and(roi_bgr, roi_bgr, mask=mask_tissue)
    tissue_only_bgr[mask_tissue == 0] = [128, 128, 128]
    pred_tissue = predict(cv2.cvtColor(tissue_only_bgr, cv2.COLOR_BGR2RGB), body_part=canon_modality)
    p_tissue = float(pred_tissue.get("anemia_probability", 0.5))
    class_tissue = str(pred_tissue.get("predicted_class", "unknown"))

    # 4. Background Only (Tissue Occluded)
    bg_only_bgr = roi_bgr.copy()
    bg_only_bgr[mask_tissue > 0] = [128, 128, 128]
    pred_bg = predict(cv2.cvtColor(bg_only_bgr, cv2.COLOR_BGR2RGB), body_part=canon_modality)
    p_bg = float(pred_bg.get("anemia_probability", 0.5))
    class_bg = str(pred_bg.get("predicted_class", "unknown"))

    # 5. Warm Illuminant (Tungsten 2800K)
    warm_bgr = roi_bgr.copy().astype(float)
    warm_bgr[:, :, 2] = np.clip(warm_bgr[:, :, 2] * 1.18, 0, 255)  # Boost Red
    warm_bgr[:, :, 0] = np.clip(warm_bgr[:, :, 0] * 0.82, 0, 255)  # Drop Blue
    warm_bgr = warm_bgr.astype(np.uint8)
    pred_warm = predict(cv2.cvtColor(warm_bgr, cv2.COLOR_BGR2RGB), body_part=canon_modality)
    p_warm = float(pred_warm.get("anemia_probability", 0.5))
    class_warm = str(pred_warm.get("predicted_class", "unknown"))

    # 6. Cool Illuminant (Overcast Daylight 6500K)
    cool_bgr = roi_bgr.copy().astype(float)
    cool_bgr[:, :, 0] = np.clip(cool_bgr[:, :, 0] * 1.18, 0, 255)  # Boost Blue
    cool_bgr[:, :, 2] = np.clip(cool_bgr[:, :, 2] * 0.82, 0, 255)  # Drop Red
    cool_bgr = cool_bgr.astype(np.uint8)
    pred_cool = predict(cv2.cvtColor(cool_bgr, cv2.COLOR_BGR2RGB), body_part=canon_modality)
    p_cool = float(pred_cool.get("anemia_probability", 0.5))
    class_cool = str(pred_cool.get("predicted_class", "unknown"))

    # 7. Optical Defocus Blur
    blur_bgr = cv2.GaussianBlur(roi_bgr, (15, 15), 3.5)
    pred_blur = predict(cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2RGB), body_part=canon_modality)
    p_blur = float(pred_blur.get("anemia_probability", 0.5))
    class_blur = str(pred_blur.get("predicted_class", "unknown"))

    # 8. Border Framing Perturbation (12% periphery stripped)
    pad_h = int(ch * 0.12)
    pad_w = int(cw * 0.12)
    border_bgr = cv2.resize(roi_bgr[pad_h : ch - pad_h, pad_w : cw - pad_w], (cw, ch))
    pred_border = predict(cv2.cvtColor(border_bgr, cv2.COLOR_BGR2RGB), body_part=canon_modality)
    p_border = float(pred_border.get("anemia_probability", 0.5))
    class_border = str(pred_border.get("predicted_class", "unknown"))

    # Assemble conditions
    def make_cond(cid, name, cat, prob, p_class, bgr, desc):
        delta = abs(prob - p_base)
        return PerturbationCondition(
            condition_id=cid,
            name=name,
            category=cat,
            probability=round(prob, 4),
            predicted_class=p_class,
            delta_prob=round(delta, 4),
            percent_shift=round(delta * 100, 1),
            thumbnail_b64=_to_base64_thumbnail(bgr),
            description=desc,
        )

    conditions = [
        make_cond(
            "baseline",
            "Target Anatomical ROI",
            "Baseline",
            p_base,
            class_base,
            roi_bgr,
            "Standard tight anatomical crop used during active screening.",
        ),
        make_cond(
            "raw_image",
            "Uncropped Full Image",
            "Spatial Context",
            p_raw,
            class_raw,
            image_bgr,
            "Full uncropped frame including room background, clothing, and full face.",
        ),
        make_cond(
            "tissue_only",
            "Isolated Tissue Only",
            "Ablation",
            p_tissue,
            class_tissue,
            tissue_only_bgr,
            "Surrounding skin and eyelid margin masked to neutral gray.",
        ),
        make_cond(
            "bg_only",
            "Background Skin Only (Tissue Occluded)",
            "Ablation",
            p_bg,
            class_bg,
            bg_only_bgr,
            "Target palpebral / nail tissue completely blacked out; only perimeter visible.",
        ),
        make_cond(
            "warm_light",
            "Warm Illuminant (+18% Red)",
            "Lighting",
            p_warm,
            class_warm,
            warm_bgr,
            "Simulates warm 2800K incandescent clinic light or evening illumination.",
        ),
        make_cond(
            "cool_light",
            "Cool Illuminant (+18% Blue)",
            "Lighting",
            p_cool,
            class_cool,
            cool_bgr,
            "Simulates cool 6500K overcast daylight or fluorescent clinic tube.",
        ),
        make_cond(
            "defocus_blur",
            "Optical Defocus Blur",
            "Sensor / Lens",
            p_blur,
            class_blur,
            blur_bgr,
            "Gaussian defocus simulating camera shake or low-cost smartphone lens.",
        ),
        make_cond(
            "border_crop",
            "Peripheral Border Trim",
            "Framing",
            p_border,
            class_border,
            border_bgr,
            "Outer 12% edge frame removed to test convolutional border anchoring.",
        ),
    ]

    # --- Compute Ranked Failure Mechanisms ---
    failure_mechanisms: list[FailureMechanism] = []

    # Mechanism 1: Background Melanin Shortcut
    # If the model still predicts the same outcome when the tissue is occluded, it is relying on skin!
    bg_shortcut_sens = round(max(0.0, 1.0 - abs(p_bg - p_base)) * 100, 1)
    if bg_shortcut_sens >= 75.0:
        bg_risk = "Critical"
    elif bg_shortcut_sens >= 50.0:
        bg_risk = "Elevated"
    elif bg_shortcut_sens >= 30.0:
        bg_risk = "Moderate"
    else:
        bg_risk = "Resilient"

    failure_mechanisms.append(
        FailureMechanism(
            id="bg_melanin_shortcut",
            name="Background Skin Melanin Bias",
            risk_level=bg_risk,
            sensitivity_score=bg_shortcut_sens,
            evidence=f"With target tissue 100% occluded, background probability remained at {p_bg*100:.1f}% (Baseline: {p_base*100:.1f}%).",
            mechanism_detail="The convolutional feature extractor partially attends to cheek / dorsal knuckle skin tones rather than strictly reading microvascular perfusion within the target tissue.",
            clinical_implication="Model accuracy will degrade when deployed in clinics with different patient Fitzpatrick phototype distributions.",
        )
    )

    # Mechanism 2: Illuminant / Color Temperature Shift
    light_delta = max(abs(p_warm - p_base), abs(p_cool - p_base))
    light_sens = round(min(100.0, light_delta * 200.0), 1)
    light_risk = "Elevated" if light_sens >= 50 else ("Moderate" if light_sens >= 25 else "Resilient")
    failure_mechanisms.append(
        FailureMechanism(
            id="illuminant_drift",
            name="Color Temperature / Spectral Drift",
            risk_level=light_risk,
            sensitivity_score=light_sens,
            evidence=f"Warm tungsten shifted probability by {abs(p_warm-p_base)*100:.1f}%; cool daylight shifted by {abs(p_cool-p_base)*100:.1f}%.",
            mechanism_detail="Shifting white balance alters optical reflectance around the 542nm and 576nm oxyhemoglobin absorption bands, confounding microvascular erythema quantification.",
            clinical_implication="Screening accuracy fluctuates depending on room lighting (incandescent bulb vs. window daylight). Optical normalization (Person 1) is mandatory.",
        )
    )

    # Mechanism 3: Contextual Spatial Leakage
    raw_delta = abs(p_raw - p_base)
    raw_sens = round(min(100.0, raw_delta * 150.0), 1)
    raw_risk = "Elevated" if raw_sens >= 50 else ("Moderate" if raw_sens >= 25 else "Resilient")
    failure_mechanisms.append(
        FailureMechanism(
            id="spatial_leakage",
            name="Macro Contextual Leakage (Uncropped Bias)",
            risk_level=raw_risk,
            evidence=f"Uncropped full image produced {p_raw*100:.1f}% probability vs {p_base*100:.1f}% on cropped ROI.",
            sensitivity_score=raw_sens,
            mechanism_detail="When uncropped, features from clothing, facial expressions, or background wallpaper infiltrate deep layer representations.",
            clinical_implication="Validates why automated YuNet and distal fingertip ROI cropping is clinically indispensable before running model heads.",
        )
    )

    # Mechanism 4: Optical Defocus & Microvascular Edge Loss
    blur_delta = abs(p_blur - p_base)
    blur_sens = round(min(100.0, blur_delta * 180.0), 1)
    blur_risk = "Elevated" if blur_sens >= 50 else ("Moderate" if blur_sens >= 25 else "Resilient")
    failure_mechanisms.append(
        FailureMechanism(
            id="lens_defocus",
            name="Optical Defocus & Motion Blur Vulnerability",
            risk_level=blur_risk,
            sensitivity_score=blur_sens,
            evidence=f"Gaussian defocus shifted probability by {blur_delta*100:.1f}%.",
            mechanism_detail="Mild lens defocus obliterates microvascular capillary loops and conjunctival vessel branching patterns.",
            clinical_implication="Confirms the critical necessity of client-side Laplacian focus gate rejection prior to inference.",
        )
    )

    # Mechanism 5: Border Framing Artifacts
    border_delta = abs(p_border - p_base)
    border_sens = round(min(100.0, border_delta * 150.0), 1)
    border_risk = "Elevated" if border_sens >= 50 else ("Moderate" if border_sens >= 25 else "Resilient")
    failure_mechanisms.append(
        FailureMechanism(
            id="border_framing",
            name="Border Framing & Vignetting Anchoring",
            risk_level=border_risk,
            sensitivity_score=border_sens,
            evidence=f"Removing 12% outer perimeter shifted probability by {border_delta*100:.1f}%.",
            mechanism_detail="Corner gradients and edge padding can subtly anchor spatial convolutional attention.",
            clinical_implication="Low-to-moderate risk; mitigated by multi-scale data augmentation during training.",
        )
    )

    # Sort mechanisms by sensitivity score descending
    failure_mechanisms.sort(key=lambda m: m.sensitivity_score, reverse=True)

    # Overall shortcut susceptibility index
    avg_sens = float(np.mean([m.sensitivity_score for m in failure_mechanisms]))
    if avg_sens >= 55.0:
        overall_status = "Elevated Shortcut Susceptibility"
        overall_class = "risk-high"
    elif avg_sens >= 35.0:
        overall_status = "Moderate Generalization Risk"
        overall_class = "risk-moderate"
    else:
        overall_status = "High Biological Grounding"
        overall_class = "risk-low"

    return {
        "modality": canon_modality,
        "baseline_probability": round(p_base * 100, 1),
        "baseline_class": class_base,
        "conditions": [c.__dict__ for c in conditions],
        "ranked_failure_mechanisms": [m.__dict__ for m in failure_mechanisms],
        "overall_shortcut_index": round(avg_sens, 1),
        "overall_status": overall_status,
        "overall_class": overall_class,
        "big_reveal_note": (
            "The Big Reveal: Your model may achieve impressive test accuracy while relying partly on "
            "a feature that won't generalize to a different clinic. This research interface provides a "
            "hypothesis-testing tool to inspect shortcut learning, not automatic proof that a specific "
            "feature caused the clinical prediction."
        ),
    }
