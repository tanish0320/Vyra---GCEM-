"""Dual-Output EfficientNet-B3 Model for Joint Anemia Classification and Hemoglobin Regression.

Architectural highlights:
- EfficientNet-B3 backbone pretrained on ImageNet
- Global Average Pooling -> 256-dim visual representation
- Multimodal feature fusion: 256 (visual) + 27 (colorimetric features) + 2 (body-part encoding) = 285-dim
- Head A: Anemia Classification (2 logits, Softmax probabilities)
- Head B: Continuous Hemoglobin Regression in g/dL with Softplus positive constraint
- Masked Joint Loss: L_total = 0.7 * L_CE + 0.3 * L_MSE (ignores samples without valid lab Hb)
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import numpy as np


class DualHeadEfficientNetB3(nn.Module):
    """Joint Classification & Hemoglobin Regression Model with Colorimetric & Anatomical Fusion."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        visual_dim: int = 256,
        feature_dim: int = 27,
        body_part_dim: int = 2,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.feature_dim = feature_dim
        self.body_part_dim = body_part_dim
        self.fused_dim = visual_dim + feature_dim + body_part_dim

        # 1. Visual Backbone
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_b3(weights=weights)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        backbone_out_features = 1536  # EfficientNet-B3 features output dimension

        # Visual projector
        self.visual_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.40),
            nn.Linear(512, visual_dim),
            nn.BatchNorm1d(visual_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
        )

        # 2. Classification Head
        self.classifier_head = nn.Sequential(
            nn.Linear(self.fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

        # 3. Hemoglobin Regression Head (Outputs strictly positive g/dL via Softplus)
        self.regression_head = nn.Sequential(
            nn.Linear(self.fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def init_regression_head(self, initial_hb_mean: float = 10.75) -> None:
        """Initialize final linear layer bias so Softplus outputs the clinical population mean."""
        linear_layer = self.regression_head[-2]
        if isinstance(linear_layer, nn.Linear):
            # Invert Softplus: x = log(exp(y) - 1). For y >= 5, x ~= y.
            val = float(np.log(np.exp(initial_hb_mean) - 1.0)) if initial_hb_mean < 15 else initial_hb_mean
            nn.init.constant_(linear_layer.bias, val)
            nn.init.normal_(linear_layer.weight, 0.0, 0.005)

    def extract_visual_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the 256-dim visual representation from image tensor."""
        feat = self.features(x)
        pooled = self.avgpool(feat)
        return self.visual_projector(pooled)

    def forward(
        self,
        images: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        body_part: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            images: Tensor of shape (B, 3, H, W)
            features: Optional Tensor of shape (B, 27) with colorimetric features
            body_part: Optional Tensor of shape (B, 2) with body part one-hot encoding
            
        Returns:
            logits: (B, 2) classification logits
            hb_estimate: (B, 1) continuous hemoglobin estimates in g/dL
        """
        batch_size = images.size(0)
        device = images.device

        # Extract visual embedding
        v_emb = self.extract_visual_embedding(images)

        # Handle optional handcrafted features
        if features is None:
            features = torch.zeros((batch_size, self.feature_dim), device=device)
        elif features.size(-1) != self.feature_dim:
            # Pad or truncate if dimensions differ
            if features.size(-1) < self.feature_dim:
                pad = torch.zeros((batch_size, self.feature_dim - features.size(-1)), device=device)
                features = torch.cat([features, pad], dim=-1)
            else:
                features = features[:, :self.feature_dim]

        # Handle optional body part encoding
        if body_part is None:
            body_part = torch.full((batch_size, self.body_part_dim), 0.5, device=device)

        # Multimodal fusion
        fused = torch.cat([v_emb, features, body_part], dim=-1)

        # Dual task outputs
        logits = self.classifier_head(fused)
        hb_estimate = self.regression_head(fused)

        return logits, hb_estimate


class MaskedDualLoss(nn.Module):
    """Weighted composite loss function handling missing hemoglobin labels safely.
    
    L_total = alpha * L_CE + beta * L_MSE
    
    When Hb is missing or NaN for a sample, it is excluded from L_MSE.
    If no samples in a batch have valid Hb, L_MSE is 0.0.
    """

    def __init__(
        self,
        alpha: float = 0.70,
        beta: float = 0.30,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        regression_loss_type: str = "smooth_l1",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.regression_loss_type = regression_loss_type
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        if regression_loss_type == "smooth_l1":
            self.reg_loss_fn = nn.SmoothL1Loss(reduction="mean", beta=1.0)
        else:
            self.reg_loss_fn = nn.MSELoss(reduction="mean")
        self.mse_loss = self.reg_loss_fn

    def forward(
        self,
        logits: torch.Tensor,
        hb_pred: torch.Tensor,
        class_targets: torch.Tensor,
        hb_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute composite loss.
        
        Args:
            logits: (B, 2) predicted class logits
            hb_pred: (B, 1) predicted Hb in g/dL
            class_targets: (B,) ground truth class labels (0 or 1)
            hb_targets: (B, 1) ground truth Hb (NaN or <0 indicates missing)
            
        Returns:
            total_loss, ce_loss, reg_loss
        """
        # 1. Classification Loss
        l_ce = self.ce_loss(logits, class_targets)

        # 2. Masked Regression Loss
        valid_mask = ~torch.isnan(hb_targets) & (hb_targets > 0.0)
        if valid_mask.sum() > 0:
            l_reg = self.reg_loss_fn(hb_pred[valid_mask], hb_targets[valid_mask])
        else:
            l_reg = torch.tensor(0.0, device=logits.device, requires_grad=True)

        l_total = self.alpha * l_ce + self.beta * l_reg
        return l_total, l_ce, l_reg
