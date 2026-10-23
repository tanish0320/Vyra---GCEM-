"""Grad-CAM and temperature-scaling utilities for AnemiaVision AI."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor, nn

from config import GRADCAM_DIR, MODELS_DIR, TRAINING_CONFIG


TEMPERATURE_SCALING_PATH = MODELS_DIR / "temperature_scaling.json"
_COLORMAP_STOPS = np.array([0.0, 0.6, 1.0], dtype=np.float32)
_COLORMAP_RGB = np.array(
    [
        [17, 92, 173],
        [248, 215, 76],
        [230, 57, 70],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class GradCAMArtifacts:
    """Files and metadata produced for one Grad-CAM render."""

    output_path: Path
    heatmap_mean: float
    heatmap_peak: float
    used_fallback: bool


class TemperatureScaler:
    """Post-hoc confidence calibration using temperature scaling."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(float(temperature), 1e-3)

    def scale(self, logits: Tensor) -> Tensor:
        """Apply calibrated temperature scaling to a logits tensor."""

        return logits / self.temperature

    __call__ = scale

    def fit(self, logits: Tensor, labels: Tensor, max_iter: int = 50) -> float:
        """Fit temperature on validation logits via negative log likelihood."""

        if logits.ndim != 2:
            raise ValueError("Expected logits with shape [N, C].")
        if labels.ndim != 1:
            raise ValueError("Expected labels with shape [N].")
        if len(logits) != len(labels):
            raise ValueError("Logits and labels must contain the same number of samples.")

        device = logits.device
        labels = labels.to(device=device, dtype=torch.long)
        log_temperature = torch.nn.Parameter(
            torch.log(torch.tensor([self.temperature], device=device, dtype=logits.dtype))
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)

        def closure() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            temperature = torch.exp(log_temperature).clamp(min=1e-3, max=10.0)
            loss = criterion(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        fitted_temperature = torch.exp(log_temperature).clamp(min=1e-3, max=10.0)
        self.temperature = float(fitted_temperature.item())
        return self.temperature

    def to_dict(self) -> dict[str, float]:
        """Serialize the temperature for JSON storage."""

        return {"temperature": float(self.temperature)}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any] | None,
        fallback_path: Path = TEMPERATURE_SCALING_PATH,
    ) -> "TemperatureScaler":
        """Load temperature metadata from a checkpoint or sidecar JSON file."""

        if checkpoint:
            candidate_keys = (
                "temperature",
                "calibration_temperature",
                "temperature_scaling",
            )
            for key in candidate_keys:
                value = checkpoint.get(key)
                if isinstance(value, (float, int)):
                    return cls(float(value))
                if isinstance(value, Mapping) and "temperature" in value:
                    return cls(float(value["temperature"]))

        if fallback_path.exists():
            try:
                payload = json.loads(fallback_path.read_text(encoding="utf-8"))
                if "temperature" in payload:
                    return cls(float(payload["temperature"]))
            except (OSError, ValueError, TypeError):
                pass

        return cls(1.0)

    def save(self, output_path: Path = TEMPERATURE_SCALING_PATH) -> None:
        """Persist the temperature so inference can reuse it later."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class GradCAM:
    """Grad-CAM implementation tailored for EfficientNet-style CNN backbones."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        """Capture activations from the target convolutional layer."""

        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[Tensor | None, ...],
        grad_output: tuple[Tensor | None, ...],
    ) -> None:
        """Capture gradients from the same target layer."""

        self.gradients = grad_output[0].detach() if grad_output and grad_output[0] is not None else None

    def generate(self, input_tensor: Tensor, class_index: int | None = None) -> tuple[np.ndarray, bool]:
        """Generate a normalized heatmap for a single input item."""

        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("Grad-CAM expects a single-item batch with shape [1, C, H, W].")

        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(input_tensor)

            if class_index is None:
                class_index = int(torch.argmax(logits, dim=1).item())

            score = logits[:, class_index]
            score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        heatmap = cam.squeeze().detach().cpu().numpy().astype(np.float32)
        heatmap, used_fallback = stabilize_heatmap(
            heatmap=heatmap,
            activations=self.activations.squeeze(0).detach().cpu().numpy().astype(np.float32),
        )
        return heatmap, used_fallback

    def close(self) -> None:
        """Remove hooks after use to avoid leaking handles."""

        self.forward_handle.remove()
        self.backward_handle.remove()


def resolve_target_layer(model: nn.Module) -> nn.Module:
    """Resolve the final convolutional layer for EfficientNet-B3."""

    if hasattr(model, "features"):
        features = model.features
        if len(features) and isinstance(features[-1], nn.Sequential):
            for module in reversed(list(features[-1].children())):
                if isinstance(module, nn.Conv2d):
                    return module
        if len(features) and hasattr(features[-1], "__getitem__"):
            try:
                candidate = features[-1][0]
                if isinstance(candidate, nn.Conv2d):
                    return candidate
            except Exception:
                pass
        for module in reversed(list(features.modules())):
            if isinstance(module, nn.Conv2d):
                return module

    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module

    raise RuntimeError("Unable to locate a convolutional layer for Grad-CAM.")


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Scale a heatmap into the [0, 1] range safely."""

    finite_heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(np.min(finite_heatmap))
    maximum = float(np.max(finite_heatmap))
    if maximum - minimum < 1e-8:
        return np.zeros_like(finite_heatmap, dtype=np.float32)
    return ((finite_heatmap - minimum) / (maximum - minimum)).astype(np.float32)


def fallback_center_heatmap(shape: tuple[int, int]) -> np.ndarray:
    """Create a soft central prior when the model activation map is empty."""

    height, width = shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    sigma_y = max(height * 0.22, 1.0)
    sigma_x = max(width * 0.22, 1.0)
    gaussian = np.exp(
        -(
            ((x_coords - center_x) ** 2) / (2 * sigma_x ** 2)
            + ((y_coords - center_y) ** 2) / (2 * sigma_y ** 2)
        )
    )
    return normalize_heatmap(gaussian.astype(np.float32))


def stabilize_heatmap(heatmap: np.ndarray, activations: np.ndarray) -> tuple[np.ndarray, bool]:
    """Handle empty or tiny Grad-CAM activations so the output stays legible."""

    normalized = normalize_heatmap(heatmap)
    target_height, target_width = normalized.shape
    used_fallback = False

    if float(np.max(normalized)) < 1e-6:
        activation_map = np.mean(np.abs(activations), axis=0)
        activation_map = cv2.resize(
            activation_map,
            (target_width, target_height),
            interpolation=cv2.INTER_CUBIC,
        )
        normalized = normalize_heatmap(activation_map)
        used_fallback = True

    if float(np.max(normalized)) < 1e-6:
        normalized = fallback_center_heatmap(normalized.shape)
        used_fallback = True

    hot_region_ratio = float(np.mean(normalized >= 0.75))
    if 0.0 < hot_region_ratio < 0.01:
        normalized = cv2.GaussianBlur(normalized, (0, 0), sigmaX=4.0, sigmaY=4.0)
        normalized = normalize_heatmap(normalized)

    return normalized, used_fallback


def apply_attention_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Map a normalized heatmap onto blue -> yellow -> red attention colors."""

    clipped = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    flat = clipped.reshape(-1)
    colorized = np.empty((flat.size, 3), dtype=np.float32)

    for channel in range(3):
        colorized[:, channel] = np.interp(flat, _COLORMAP_STOPS, _COLORMAP_RGB[:, channel])

    return colorized.reshape((*clipped.shape, 3)).astype(np.uint8)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    """Blend the attention heatmap onto the resized RGB image."""

    colored = apply_attention_colormap(heatmap)
    overlay = (image_rgb.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha).clip(0, 255)
    return overlay.astype(np.uint8)


def build_side_by_side_image(original_rgb: np.ndarray, overlay_rgb: np.ndarray) -> np.ndarray:
    """Create one comparison image with labels for the original and heatmap."""

    original = Image.fromarray(original_rgb)
    overlay = Image.fromarray(overlay_rgb)
    label_band_height = 38
    padding = 16
    canvas_width = original.width * 2 + padding * 3
    canvas_height = original.height + label_band_height + padding * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, canvas_width, label_band_height + padding), fill=(245, 248, 250))
    draw.text((padding, 12), "Original", fill=(13, 27, 42), font=font)
    draw.text((original.width + padding * 2, 12), "Grad-CAM Overlay", fill=(13, 27, 42), font=font)

    canvas.paste(original, (padding, label_band_height + padding))
    canvas.paste(overlay, (original.width + padding * 2, label_band_height + padding))
    return np.array(canvas)


def generate_and_save_gradcam(
    model: nn.Module,
    image_path: Path,
    input_tensor: Tensor,
    predicted_class_index: int,
    output_dir: Path = GRADCAM_DIR,
) -> GradCAMArtifacts:
    """Generate and save a labeled side-by-side Grad-CAM comparison image."""

    output_dir.mkdir(parents=True, exist_ok=True)
    target_layer = resolve_target_layer(model)
    gradcam = GradCAM(model=model, target_layer=target_layer)

    try:
        heatmap, used_fallback = gradcam.generate(
            input_tensor=input_tensor,
            class_index=predicted_class_index,
        )
    finally:
        gradcam.close()

    with Image.open(image_path) as image:
        original_rgb = image.convert("RGB").resize(TRAINING_CONFIG.image_size)

    original_array = np.array(original_rgb)
    overlay_array = overlay_heatmap(original_array, heatmap)
    comparison = build_side_by_side_image(original_array, overlay_array)

    output_path = output_dir / f"{image_path.stem}_gradcam_{uuid.uuid4().hex[:8]}.png"
    Image.fromarray(comparison).save(output_path)

    return GradCAMArtifacts(
        output_path=output_path,
        heatmap_mean=float(np.mean(heatmap)),
        heatmap_peak=float(np.max(heatmap)),
        used_fallback=used_fallback,
    )
