"""Production-oriented prediction helpers for AnemiaVision AI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image

from config import BEST_CHECKPOINT_PATH, CLASS_TO_IDX, IDX_TO_CLASS, TRAINING_CONFIG
from dataset import build_transforms, create_dataloaders
from gradcam import TemperatureScaler, generate_and_save_gradcam
from image_validator import validate_image
from model import latest_available_checkpoint, load_checkpoint


_DEFAULT_PREDICTOR: "AnemiaPredictor | None" = None


def autocast_context():
    """Return a version-compatible autocast context manager for CUDA inference."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=TRAINING_CONFIG.use_amp)
    return torch.cuda.amp.autocast(enabled=TRAINING_CONFIG.use_amp)


def normalize_prediction_label(predicted_class: str) -> str:
    """Convert internal class names into UI-ready labels."""

    token = predicted_class.strip().lower()
    if token.startswith("non"):
        return "Non-Anemic"
    return "Anemic"


def compute_risk_metadata(prediction: str, confidence: float) -> tuple[str, str]:
    """Map the model decision into a human-readable risk label and explanation."""

    confidence_percent = confidence * 100.0

    if prediction == "Non-Anemic":
        if confidence_percent > 85.0:
            return (
                "Low Risk",
                "The scan looks reassuring, with no strong AI signs of anemia. Routine monitoring is appropriate.",
            )
        if confidence_percent >= 70.0:
            return (
                "Low-Medium Risk",
                "The scan leans non-anemic, but confidence is moderate. A repeat image or follow-up screening is sensible.",
            )
        return (
            "Uncertain",
            "The model does not see strong anemia features, but confidence is limited. A clinical blood test is recommended if symptoms are present.",
        )

    if confidence_percent > 85.0:
        return (
            "High Risk",
            "The AI detected strong anemia-associated patterns. Please arrange a prompt medical review and confirm with a CBC or hemoglobin test.",
        )
    if confidence_percent >= 70.0:
        return (
            "Medium Risk",
            "The scan shows meaningful anemia-related cues. A doctor visit and confirmatory lab testing are recommended.",
        )
    return (
        "Possible Anemia",
        "The model leans toward anemia, but confidence is limited. Clinical confirmation with a blood test is recommended.",
    )


def medical_advice_lines(prediction: str) -> list[str]:
    """Return patient-facing follow-up guidance for the predicted class."""

    if prediction == "Anemic":
        return [
            "Please consult a doctor or clinic soon for a full medical assessment.",
            "Consider a CBC blood test or hemoglobin test to confirm the screening result.",
            "Support iron intake with iron-rich foods if medically appropriate, such as leafy greens, beans, lentils, fortified cereals, and lean meats.",
            "Seek urgent care sooner if there is dizziness, fainting, chest pain, shortness of breath, or severe weakness.",
        ]

    return [
        "Results look healthy at this screening level, with no strong visual signs of anemia detected.",
        "Continue a balanced diet that supports iron, folate, vitamin B12, and hydration.",
        "If fatigue, paleness, breathlessness, or other symptoms continue, arrange a clinical blood test even after a non-anemic result.",
    ]


def medical_advice_for_prediction(prediction: str) -> str:
    """Return the same advice as one concise paragraph for APIs and reports."""

    return " ".join(medical_advice_lines(prediction))


def estimate_hemoglobin_range(prediction: str, confidence: float) -> str:
    """Provide a rough, non-diagnostic hemoglobin estimate band."""

    if prediction == "Anemic":
        if confidence >= 0.85:
            return "7-9 g/dL (estimated)"
        if confidence >= 0.70:
            return "9-11 g/dL (estimated)"
        return "10-12 g/dL (estimated)"

    if confidence >= 0.85:
        return "12-15 g/dL (estimated)"
    if confidence >= 0.70:
        return "11-14 g/dL (estimated)"
    return "10.5-13 g/dL (estimated)"


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction response for the CLI, API, and Flask app."""

    prediction: str | None
    confidence: float
    anemic_probability: float
    non_anemic_probability: float
    risk_level: str | None
    risk_explanation: str
    gradcam_path: str | None
    medical_advice: str
    hemoglobin_estimate: str | None
    processing_time_ms: int
    image_quality: str | None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def predicted_class(self) -> str:
        """Backward-compatible alias used by the Flask app."""

        if not self.prediction:
            return ""
        return self.prediction.lower()

    @property
    def predicted_index(self) -> int:
        """Backward-compatible alias used by older code paths."""

        if self.prediction == "Anemic":
            return CLASS_TO_IDX["anemic"]
        if self.prediction == "Non-Anemic":
            return CLASS_TO_IDX["non-anemic"]
        return -1

    @property
    def probabilities(self) -> dict[str, float]:
        """Backward-compatible probability mapping."""

        return {
            "anemic": self.anemic_probability,
            "non-anemic": self.non_anemic_probability,
        }

    def to_dict(self) -> dict[str, object]:
        """Convert the result into an API-friendly dictionary."""

        return asdict(self)

    def to_json(self) -> str:
        """Serialize the prediction as formatted JSON."""

        return json.dumps(self.to_dict(), indent=2)


class AnemiaPredictor:
    """Load a checkpoint once and reuse it for repeated predictions."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        auto_calibrate: bool = True,
    ) -> None:
        self.device = TRAINING_CONFIG.device
        self.checkpoint_path = checkpoint_path or latest_available_checkpoint() or BEST_CHECKPOINT_PATH
        self.model, self.checkpoint = load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            pretrained_fallback=False,
        )
        image_size = TRAINING_CONFIG.image_size
        config_snapshot = self.checkpoint.get("config", {}) if isinstance(self.checkpoint, dict) else {}
        training_snapshot = (
            config_snapshot.get("training", {}) if isinstance(config_snapshot, dict) else {}
        )
        if isinstance(training_snapshot, dict):
            checkpoint_image_size = training_snapshot.get("image_size")
            if isinstance(checkpoint_image_size, (list, tuple)) and len(checkpoint_image_size) == 2:
                image_size = tuple(checkpoint_image_size)

        _, self.eval_transform, _ = build_transforms(image_size=image_size)
        self.temperature_scaler = TemperatureScaler.from_checkpoint(self.checkpoint)

        if auto_calibrate and abs(self.temperature_scaler.temperature - 1.0) < 1e-6:
            self._fit_temperature_scaler()

    def _fit_temperature_scaler(self) -> None:
        """Fit temperature scaling on the validation split when metadata is missing."""

        try:
            dataloaders = create_dataloaders(
                batch_size=TRAINING_CONFIG.batch_size,
                num_workers=0,
                show_statistics=False,
            )
        except Exception:
            return

        logits_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in dataloaders.valid_loader:
                images = images.to(self.device, non_blocking=TRAINING_CONFIG.pin_memory)
                with autocast_context():
                    logits = self.model(images)
                logits_batches.append(logits.detach().float().cpu())
                label_batches.append(labels.detach().cpu())

        if not logits_batches:
            return

        try:
            all_logits = torch.cat(logits_batches, dim=0)
            all_labels = torch.cat(label_batches, dim=0)
            self.temperature_scaler.fit(all_logits, all_labels)
            self.temperature_scaler.save()
        except Exception:
            self.temperature_scaler = TemperatureScaler(1.0)

    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        """Load one image from disk and convert it into a model-ready tensor."""

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.eval_transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict_image(
        self,
        image_path: str | Path,
        patient_info: dict[str, object] | None = None,
        save_gradcam: bool = True,
    ) -> PredictionResult:
        """Run the full prediction workflow for a single image file."""

        started_at = perf_counter()
        resolved_path = Path(image_path).expanduser().resolve()
        validation = validate_image(resolved_path)

        if not validation.is_valid:
            return PredictionResult(
                prediction=None,
                confidence=0.0,
                anemic_probability=0.0,
                non_anemic_probability=0.0,
                risk_level=None,
                risk_explanation="Inference was not started because the uploaded image failed validation.",
                gradcam_path=None,
                medical_advice="Please retake the image with better quality before attempting another scan.",
                hemoglobin_estimate=None,
                processing_time_ms=int(round((perf_counter() - started_at) * 1000)),
                image_quality=validation.image_quality,
                error=validation.error,
                warnings=validation.warnings,
            )

        input_tensor = self.preprocess_image(resolved_path)

        self.model.eval()
        with torch.no_grad():
            with autocast_context():
                raw_logits = self.model(input_tensor)
            calibrated_logits = self.temperature_scaler.scale(raw_logits.float())
            probabilities = torch.softmax(calibrated_logits, dim=1).squeeze(0).cpu()

        predicted_index = int(torch.argmax(probabilities).item())
        prediction = normalize_prediction_label(IDX_TO_CLASS[predicted_index])
        confidence = float(probabilities[predicted_index].item())
        anemic_probability = float(probabilities[CLASS_TO_IDX["anemic"]].item())
        non_anemic_probability = float(probabilities[CLASS_TO_IDX["non-anemic"]].item())
        risk_level, risk_explanation = compute_risk_metadata(prediction, confidence)

        warnings = list(validation.warnings)
        gradcam_path: str | None = None

        if save_gradcam:
            try:
                gradcam_artifacts = generate_and_save_gradcam(
                    model=self.model,
                    image_path=resolved_path,
                    input_tensor=input_tensor,
                    predicted_class_index=predicted_index,
                )
                gradcam_path = str(gradcam_artifacts.output_path)
                if gradcam_artifacts.used_fallback:
                    warnings.append(
                        "Grad-CAM used a low-activation fallback because the attention region was extremely small."
                    )
            except Exception as exc:
                warnings.append(f"Grad-CAM generation was unavailable: {exc}")

        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            anemic_probability=anemic_probability,
            non_anemic_probability=non_anemic_probability,
            risk_level=risk_level,
            risk_explanation=risk_explanation,
            gradcam_path=gradcam_path,
            medical_advice=medical_advice_for_prediction(prediction),
            hemoglobin_estimate=estimate_hemoglobin_range(prediction, confidence),
            processing_time_ms=int(round((perf_counter() - started_at) * 1000)),
            image_quality=validation.image_quality,
            error=None,
            warnings=warnings,
        )


def predict(image_path: str | Path, patient_info: dict[str, object] | None = None) -> dict[str, object]:
    """Convenience wrapper that returns the Stage 4 prediction dictionary."""

    global _DEFAULT_PREDICTOR

    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = AnemiaPredictor()

    return _DEFAULT_PREDICTOR.predict_image(
        image_path=image_path,
        patient_info=patient_info,
        save_gradcam=True,
    ).to_dict()


def main() -> None:
    """Command-line prediction entry point."""

    import argparse

    parser = argparse.ArgumentParser(description="Predict anemia status from a single image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path.")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM generation.")
    args = parser.parse_args()

    predictor = AnemiaPredictor(
        checkpoint_path=Path(args.checkpoint).resolve() if args.checkpoint else None
    )
    result = predictor.predict_image(args.image_path, save_gradcam=not args.no_gradcam)
    print(result.to_json())


if __name__ == "__main__":
    main()
