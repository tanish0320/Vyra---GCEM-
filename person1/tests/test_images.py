from io import BytesIO
import json
import math
import struct
import zlib

import numpy as np
from PIL import Image
import pytest

from vyra_preprocessing import preprocess_image, preprocess_with_details
from vyra_preprocessing.artifacts import export_preview, result_json
from vyra_preprocessing.image_io import InputError, load_image, repair_png_profile_crc


def color_image(h=80, w=100):
    return np.full((h, w, 3), (170, 95, 100), dtype=np.uint8)


@pytest.mark.parametrize("dimension,valid", [(49, False), (50, True)])
def test_resolution_checked_before_resize(dimension, valid):
    result = preprocess_image(color_image(dimension, 70), "eye", pre_cropped=True)
    assert result.is_valid is valid
    if not valid:
        assert any("IMAGE_TOO_SMALL" in item for item in result.quality_warnings)
        assert result.feature_vector.shape == (27,)


@pytest.mark.parametrize("image,body,code", [
    (np.zeros((80, 80), dtype=np.uint8), "eye", "GRAYSCALE_INPUT"),
    (np.ones((80, 80, 3), dtype=np.float32), "eye", "UNSUPPORTED_DTYPE"),
    (np.zeros((80, 80, 4), dtype=np.uint8), "eye", "INVALID_CHANNELS"),
    (np.zeros((0, 80, 3), dtype=np.uint8), "eye", "INVALID_CHANNELS"),
    (color_image(), "tongue", "UNSUPPORTED_BODY_PART"),
    (color_image(), None, "UNSUPPORTED_BODY_PART"),
])
def test_invalid_inputs_have_actionable_status(image, body, code):
    result = preprocess_image(image, body, pre_cropped=True)
    assert not result.is_valid
    assert any(code in warning for warning in result.quality_warnings)
    assert np.isfinite(result.feature_vector).all()


def test_missing_and_corrupt_files(tmp_path):
    assert not preprocess_image(tmp_path / "missing.png", "eye").is_valid
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    assert not preprocess_image(path, "eye").is_valid


def test_grayscale_pil_rejected():
    assert not preprocess_image(Image.new("L", (80, 80)), "eye", pre_cropped=True).is_valid


def test_opaque_full_frame_requires_roi():
    details = preprocess_with_details(color_image(), "eye")
    assert not details.result.is_valid and details.metadata["error_code"] == "ROI_REQUIRED"


def test_transparency_is_not_tissue_or_glare():
    rgba = np.full((100, 100, 4), 255, dtype=np.uint8)
    rgba[:, :, 3] = 0
    rgba[20:80, 20:80] = [170, 95, 100, 255]
    details = preprocess_with_details(Image.fromarray(rgba), "nail")
    assert details.result.is_valid
    assert details.metadata["roi_bbox"] == [20, 20, 80, 80]
    assert details.result.glare_fraction == 0
    assert details.result.normalized_image_rgb.shape == (60, 60, 3)
    rgba[:, :, :3][rgba[:, :, 3] == 0] = [10, 245, 1]
    other = preprocess_image(Image.fromarray(rgba), "nail")
    np.testing.assert_array_equal(details.result.feature_vector, other.feature_vector)


def test_white_background_mask_prevents_false_glare():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[25:75, 25:75] = [170, 95, 100]
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:75, 25:75] = True
    result = preprocess_image(image, "eye", roi_mask=mask)
    assert result.is_valid and result.glare_fraction == 0
    assert not preprocess_image(image, "eye", pre_cropped=True).is_valid


def test_irregular_mask_does_not_expose_extension_texture_to_model():
    image = np.random.default_rng(71).integers(60, 180, (100, 100, 3), dtype=np.uint8)
    yy, xx = np.indices((100, 100))
    mask = ((xx - 50) ** 2 + (yy - 50) ** 2) < 35 ** 2
    details = preprocess_with_details(image, "eye", roi_mask=mask)
    assert details.result.is_valid and (~details.roi_mask).any()
    assert np.all(details.result.normalized_image_rgb[~details.roi_mask] == 127)


def test_excessive_glare_is_invalid():
    result = preprocess_image(np.full((80, 80, 3), 255, dtype=np.uint8), "nail", pre_cropped=True)
    assert not result.is_valid and result.glare_fraction == 1


@pytest.mark.parametrize("kwargs", [
    {"roi_mask": np.zeros((80, 100), dtype=bool)},
    {"roi_mask": np.ones((2, 2), dtype=bool)},
    {"roi_mask": np.ones((80, 100), dtype=float)},
    {"roi_bbox": (-1, 0, 10, 10)},
    {"roi_bbox": (5, 5, 4, 4)},
    {"roi_bbox": (0, 0, 20, 20), "roi_mask": np.ones((80, 100), dtype=bool)},
])
def test_invalid_rois_rejected(kwargs):
    assert not preprocess_image(color_image(), "eye", **kwargs).is_valid


def test_exif_orientation_is_applied_before_roi_coordinates(tmp_path):
    image = Image.fromarray(color_image(80, 60))
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)
    details = preprocess_with_details(path, "eye", roi_bbox=(0, 0, 80, 60))
    assert details.result.is_valid
    assert details.result.raw_image_rgb.shape == (60, 80, 3)
    assert details.metadata["exif_orientation"] == 6


def test_in_memory_pil_exif_is_preserved():
    image = Image.fromarray(color_image(80, 60))
    image.getexif()[274] = 6
    details = preprocess_with_details(image, "eye", roi_bbox=(0, 0, 80, 60))
    assert details.result.is_valid and details.metadata["exif_orientation"] == 6
    assert details.result.raw_image_rgb.shape == (60, 80, 3)


def test_skin_mask_required_for_ita_and_json_has_null(tmp_path):
    details = preprocess_with_details(color_image(), "eye", pre_cropped=True)
    assert math.isnan(details.result.ita_angle)
    data = result_json(details)
    assert data["ita_angle"] is None and data["ita_category"] == "Unknown"
    json.dumps(data, allow_nan=False)
    export_preview(details, tmp_path / "preview")
    assert json.loads((tmp_path / "preview" / "metadata.json").read_text())["ita_angle"] is None


def test_explicit_surrounding_skin_ita():
    image = np.full((100, 100, 3), (190, 140, 100), dtype=np.uint8)
    image[20:80, 20:80] = [170, 95, 100]
    skin = np.ones((100, 100), dtype=bool)
    skin[20:80, 20:80] = False
    result = preprocess_image(image, "eye", roi_bbox=(20, 20, 80, 80), skin_mask=skin)
    assert result.is_valid and math.isfinite(result.ita_angle) and result.ita_category != "Unknown"


def _png_chunk(tag, body, corrupt=False):
    crc = zlib.crc32(tag + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc ^ int(corrupt))


def test_only_profile_crc_is_repaired_without_changing_pixels(tmp_path):
    source = BytesIO()
    original = color_image()
    Image.fromarray(original).save(source, format="PNG")
    data = source.getvalue()
    ihdr_end = 8 + 12 + struct.unpack_from(">I", data, 8)[0]
    profile = b"fixture\x00\x00" + zlib.compress(b"synthetic metadata, not clinical data")
    broken = data[:ihdr_end] + _png_chunk(b"iCCP", profile, True) + data[ihdr_end:]
    path = tmp_path / "profile.png"
    path.write_bytes(broken)
    loaded = load_image(path)
    assert loaded.metadata["png_profile_crc_repaired"]
    np.testing.assert_array_equal(loaded.rgb, original)
    assert path.read_bytes() == broken
    damaged = bytearray(data)
    damaged[-5] ^= 1
    with pytest.raises(InputError, match="checksum"):
        repair_png_profile_crc(bytes(damaged))
