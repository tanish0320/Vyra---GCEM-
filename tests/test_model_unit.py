"""Unit tests for the DualHeadEfficientNetB3 model and MaskedDualLoss."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest
from src.models.dual_head_model import DualHeadEfficientNetB3, MaskedDualLoss


def test_model_forward_pass_shapes():
    model = DualHeadEfficientNetB3(pretrained=False, freeze_backbone=False)
    model.eval()

    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_features = torch.randn(batch_size, 27)
    dummy_body_part = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    logits, hb = model(dummy_images, dummy_features, dummy_body_part)

    assert logits.shape == (batch_size, 2), f"Expected shape {(batch_size, 2)}, got {logits.shape}"
    assert hb.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {hb.shape}"
    assert (hb >= 0.0).all(), "Softplus constraint violated: Hb must be >= 0.0"


def test_model_optional_inputs():
    model = DualHeadEfficientNetB3(pretrained=False)
    model.eval()

    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)

    # Test with None for features and body_part
    logits, hb = model(dummy_images, None, None)
    assert logits.shape == (batch_size, 2)
    assert hb.shape == (batch_size, 1)


def test_masked_dual_loss_with_missing_hb():
    criterion = MaskedDualLoss(alpha=0.7, beta=0.3)

    logits = torch.randn(4, 2, requires_grad=True)
    hb_pred = torch.tensor([[11.5], [12.0], [8.5], [9.2]], requires_grad=True)
    class_targets = torch.tensor([0, 1, 1, 0])

    # Sample 0 and 2 have valid Hb; sample 1 has NaN; sample 3 has -1.0 (missing)
    hb_targets = torch.tensor([[11.2], [float('nan')], [8.9], [-1.0]])

    total_loss, ce_loss, mse_loss = criterion(logits, hb_pred, class_targets, hb_targets)

    assert total_loss.item() > 0.0
    assert not torch.isnan(total_loss)
    assert not torch.isnan(mse_loss)
    total_loss.backward()
    assert logits.grad is not None
    assert hb_pred.grad is not None


def test_masked_dual_loss_all_missing_hb():
    criterion = MaskedDualLoss(alpha=0.7, beta=0.3)

    logits = torch.randn(2, 2, requires_grad=True)
    hb_pred = torch.tensor([[10.5], [13.0]], requires_grad=True)
    class_targets = torch.tensor([0, 1])
    hb_targets = torch.tensor([[float('nan')], [float('nan')]])

    total_loss, ce_loss, mse_loss = criterion(logits, hb_pred, class_targets, hb_targets)

    assert total_loss.item() > 0.0
    assert mse_loss.item() == 0.0
    total_loss.backward()
    assert logits.grad is not None


if __name__ == "__main__":
    test_model_forward_pass_shapes()
    test_model_optional_inputs()
    test_masked_dual_loss_with_missing_hb()
    test_masked_dual_loss_all_missing_hb()
    print("ALL MODEL UNIT TESTS PASSED!")
