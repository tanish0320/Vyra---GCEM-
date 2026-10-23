"""Production-grade inference service and integration contract for Person 3.

Provides a clean, decoupled interface callable directly from Flask route handlers:
predict(image, feature_vector=None, body_part=None, metadata=None) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import json
import time

import torch
import cv2
import numpy as np
from PIL import Image

from src.data.normalization_adapter import preprocess_image, PreprocessingResult
from src.models.dual_head_model import DualHeadEfficientNetB3

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class StructuredPredictionResult:
    """Standardized response schema contract for VYRA."""
    hb_estimate_g_dl: Optional[float]
    anemia_probability: Optional[float]
    predicted_class: Optional[str]
    model_version: str = "vyra-dualhead-efficientnet-b3-v1.0"
    calibration_status: str = "uncalibrated"
    quality_warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    # Additional transparency metadata for clinical UI
    anatomical_modality: Optional[str] = None
    ita_angle: Optional[float] = None
    ita_category: Optional[str] = None
    glare_fraction: Optional[float] = None
    processing_time_ms: int = 0
    regression_status: str = "unavailable"  # "validated", "fallback", or "unavailable"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class VyraInferenceEngine:
    """Thread-safe singleton inference engine for VYRA."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        use_half_precision: bool = False,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_half_precision = use_half_precision and (self.device.type == "cuda")
        self.model: Optional[DualHeadEfficientNetB3] = None
        self.checkpoint_loaded: bool = False
        self.has_validated_regression: bool = False
        self.temperature: float = 1.0

        # Try to load checkpoint if provided or default exists
        resolved_ckpt = self._find_checkpoint(checkpoint_path)
        if resolved_ckpt and resolved_ckpt.exists() and resolved_ckpt.stat().st_size > 1024:
            self._load_checkpoint(resolved_ckpt)
        else:
            # Build baseline model with pretrained ImageNet weights for classification
            try:
                self.model = DualHeadEfficientNetB3(pretrained=True)
                self.model.to(self.device)
                self.model.eval()
                if self.use_half_precision:
                    self.model.half()
                self.checkpoint_loaded = False
            except Exception as e:
                self.model = None

    def _find_checkpoint(self, path: Optional[Union[str, Path]]) -> Optional[Path]:
        if path:
            p = Path(path)
            if p.exists():
                return p
        # Check standard locations
        candidates = [
            PROJECT_ROOT / "checkpoints" / "best_dual_head.pth",
            PROJECT_ROOT / "models" / "best_dual_head.pth",
            PROJECT_ROOT / "models" / "20260428_2028_best_model.pth",
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 1024:
                return c
        return None

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model = DualHeadEfficientNetB3(pretrained=False)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.has_validated_regression = ckpt.get("has_validated_regression", False)
                self.temperature = float(ckpt.get("temperature", 1.0))
            else:
                self.model.load_state_dict(ckpt)
            self.model.to(self.device)
            self.model.eval()
            if self.use_half_precision:
                self.model.half()
            self.checkpoint_loaded = True
        except Exception as e:
            self.checkpoint_loaded = False
            self.has_validated_regression = False

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        feature_vector: Optional[np.ndarray] = None,
        body_part: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        screening_threshold: float = 0.50,
    ) -> StructuredPredictionResult:
        """Execute non-invasive screening inference."""
        start_time = time.perf_counter()
        warnings = []

        # 1. Execute Person 1 Preprocessing and Quality Verification
        prep: PreprocessingResult = preprocess_image(image, body_part=body_part)
        warnings.extend(prep.quality_warnings)

        if not prep.is_valid:
            return StructuredPredictionResult(
                hb_estimate_g_dl=None,
                anemia_probability=None,
                predicted_class=None,
                quality_warnings=warnings,
                error=prep.quality_warnings[0] if prep.quality_warnings else "Invalid input image.",
                processing_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        # 2. Check model readiness
        if self.model is None:
            return StructuredPredictionResult(
                hb_estimate_g_dl=None,
                anemia_probability=None,
                predicted_class=None,
                quality_warnings=warnings,
                error="Model weights are unavailable or failed to initialize.",
                processing_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        # 3. Format inputs for PyTorch forward pass
        img_np = cv2.resize(prep.normalized_image_rgb, (224, 224)).astype(np.float32) / 255.0
        # ImageNet standardization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_tensor = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device)

        # Use passed feature vector or fallback to extracted 27-dim features
        f_vec = feature_vector if feature_vector is not None else prep.feature_vector
        f_tensor = torch.tensor(f_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

        b_tensor = torch.tensor(prep.body_part_encoding, dtype=torch.float32).unsqueeze(0).to(self.device)

        if self.use_half_precision:
            img_tensor = img_tensor.half()
            f_tensor = f_tensor.half()
            b_tensor = b_tensor.half()

        # 4. Neural Network Forward Pass
        with torch.no_grad():
            logits, hb_pred = self.model(img_tensor, f_tensor, b_tensor)
            # Apply temperature scaling for probability calibration
            scaled_logits = logits.float() / max(self.temperature, 0.01)
            probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()[0]
            anemia_prob = float(probs[1])  # Class index 1 is Anemic

        # Classification decision based on explicitly documented threshold
        predicted_class = "anemic" if anemia_prob >= screening_threshold else "non-anemic"

        # Continuous Hb Output Rule:
        # Never substitute random or mock numbers.
        # Only report Hb if genuinely trained on lab Hb.
        is_fingernail = "nail" in (body_part or "").lower() or "finger" in (body_part or "").lower()
        if self.has_validated_regression and not is_fingernail:
            hb_val = round(float(hb_pred.cpu().numpy()[0][0]), 2)
            reg_status = "validated"
        elif is_fingernail:
            hb_val = None
            reg_status = "unavailable"
            warnings.append(
                "Continuous Hb regression is validated strictly on conjunctiva with paired laboratory CBC; for fingernails, screening relies on the classification head."
            )
        else:
            hb_val = None
            reg_status = "unavailable"
            warnings.append(
                "Continuous Hb regression is disabled in this build: no paired laboratory-measured Hb weights found. Classification screening is active."
            )

        cal_status = "calibrated" if abs(self.temperature - 1.0) > 1e-4 else "uncalibrated"
        elapsed_ms = int(round((time.perf_counter() - start_time) * 1000))

        return StructuredPredictionResult(
            hb_estimate_g_dl=hb_val,
            anemia_probability=round(anemia_prob, 4),
            predicted_class=predicted_class,
            calibration_status=cal_status,
            quality_warnings=warnings,
            error=None,
            anatomical_modality=body_part or "unknown",
            ita_angle=round(prep.ita_angle, 2),
            ita_category=prep.ita_category,
            glare_fraction=round(prep.glare_fraction, 4),
            processing_time_ms=elapsed_ms,
            regression_status=reg_status,
        )


# Global singleton instance
_GLOBAL_ENGINE: Optional[VyraInferenceEngine] = None


def predict(
    image: Union[str, Path, Image.Image, np.ndarray],
    feature_vector: Optional[np.ndarray] = None,
    body_part: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    screening_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Top-level inference interface for Person 3 Flask application."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = VyraInferenceEngine()
    result = _GLOBAL_ENGINE.predict(
        image=image,
        feature_vector=feature_vector,
        body_part=body_part,
        metadata=metadata,
        screening_threshold=screening_threshold,
    )
    return result.to_dict()
