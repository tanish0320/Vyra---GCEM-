from dataclasses import fields
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from vyra_preprocessing import FEATURE_NAMES, PreprocessingResult, extract_27_colorimetric_features, preprocess_image


@pytest.mark.parametrize("shape", [(53, 79), (224, 224), (151, 97)])
def test_all_27_features_match_real_person2_code(person2_reference, shape):
    image = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
    expected = person2_reference.extract_27_colorimetric_features(image)
    actual = extract_27_colorimetric_features(image)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (27,) and actual.dtype == np.float32


@pytest.mark.parametrize("color", [(0, 0, 0), (170, 95, 100), (255, 255, 255)])
def test_constant_color_features_match_reference(person2_reference, color):
    image = np.full((64, 80, 3), color, dtype=np.uint8)
    np.testing.assert_array_equal(extract_27_colorimetric_features(image), person2_reference.extract_27_colorimetric_features(image))


def test_result_fields_are_unchanged(person2_reference):
    assert [f.name for f in fields(PreprocessingResult)] == [f.name for f in fields(person2_reference.PreprocessingResult)]
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 27


def test_compatibility_wrapper_imports():
    path = Path(__file__).parents[1] / "integration" / "normalization_adapter.py"
    spec = importlib.util.spec_from_file_location("consumer_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = np.full((60, 80, 3), (170, 95, 100), dtype=np.uint8)
    result = module.preprocess_image(image, "eye", pre_cropped=True)
    assert result.is_valid
    assert result.feature_vector.shape == (27,)
    assert result.body_part_encoding.tolist() == [1.0, 0.0]


def test_deterministic_output_and_no_input_mutation():
    image = np.random.default_rng(1).integers(40, 190, (80, 100, 3), dtype=np.uint8)
    original = image.copy()
    first = preprocess_image(image, "nail", pre_cropped=True)
    second = preprocess_image(image, "fingernail", pre_cropped=True)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(first.normalized_image_rgb, second.normalized_image_rgb)
    np.testing.assert_array_equal(first.feature_vector, second.feature_vector)
    assert first.raw_image_rgb.dtype == first.normalized_image_rgb.dtype == np.uint8
    assert math.isnan(first.ita_angle) and first.ita_category == "Unknown"

