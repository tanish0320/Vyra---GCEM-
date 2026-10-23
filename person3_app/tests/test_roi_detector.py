"""Unit tests for anatomical ROI detection and retake rejection in person3_app."""

import sys
from pathlib import Path
APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import cv2
import numpy as np
import pytest
from PIL import Image

from roi_detector import (
    detect_and_crop_roi,
    detect_and_crop_conjunctiva,
    detect_and_crop_fingernail,
    RETAKE_GUIDANCE_CONJUNCTIVA,
    RETAKE_GUIDANCE_FINGERNAIL,
)


def test_blank_image_fails_with_retake_guidance():
    """Blank or noise images should be rejected with helpful retake instructions."""
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    res = detect_and_crop_roi(blank, modality="conjunctiva")
    assert res.success is False
    assert res.detection_type == "detection_failed"
    assert res.retake_guidance == RETAKE_GUIDANCE_CONJUNCTIVA

    res_nail = detect_and_crop_roi(blank, modality="fingernail")
    assert res_nail.success is False
    assert res_nail.detection_type == "detection_failed"
    assert res_nail.retake_guidance == RETAKE_GUIDANCE_FINGERNAIL


def test_pre_cropped_macro_conjunctiva_accepted():
    """Close-up macro conjunctiva images (like CP-AnemiC) should pass without cropping."""
    cp_path = Path("data/raw/cp_anemic/CP-AnemiC/Anemic/Image_001.png")
    if cp_path.exists():
        res = detect_and_crop_roi(cp_path, modality="conjunctiva")
        assert res.success is True
        assert res.detection_type == "pre_cropped_macro"
        assert res.cropped_image_rgb is not None


def test_pre_cropped_macro_fingernail_accepted():
    """Close-up macro fingernail images (like Ghana fingernails) should pass without cropping."""
    gh_path = Path("data/raw/ghana_fingernails/Fingernails/Anemic-Fin-008 (10).png")
    if gh_path.exists():
        res = detect_and_crop_roi(gh_path, modality="fingernail")
        assert res.success is True
        assert res.detection_type == "pre_cropped_macro"
        assert res.cropped_image_rgb is not None


def test_full_face_auto_crops_conjunctiva():
    """A full-face photo must automatically detect eyes and crop the conjunctiva."""
    face_path = Path(r"C:\Users\urbra\Downloads\WhatsApp Image 2026-10-23 at 16.40.59.jpeg")
    if face_path.exists():
        res = detect_and_crop_roi(face_path, modality="Eye Conjunctiva")
        assert res.success is True
        assert res.detection_type == "auto_cropped"
        assert res.roi_bbox is not None
        assert res.cropped_image_rgb.shape[0] >= 100
        assert res.cropped_image_rgb.shape[1] >= 100


def test_full_face_rejected_for_fingernail_modality():
    """A full face photo uploaded under fingernail modality must be rejected with advice to switch modality."""
    face_path = Path(r"C:\Users\urbra\Downloads\WhatsApp Image 2026-10-23 at 16.40.59.jpeg")
    if face_path.exists():
        res = detect_and_crop_roi(face_path, modality="Fingernail")
        assert res.success is False
        assert "face was detected" in res.message.lower()
        assert "Eye Conjunctiva" in res.retake_guidance
