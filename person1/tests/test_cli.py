import json

import numpy as np
from PIL import Image

from vyra_preprocessing.cli import main


def test_cli_image_valid_then_invalid_has_no_stale_normalized_preview(tmp_path, capsys):
    source = tmp_path / "source.png"
    destination = tmp_path / "preview"
    Image.fromarray(np.full((80, 80, 3), [170, 95, 100], dtype=np.uint8)).save(source)
    assert main(["image", str(source), "--body-part", "eye", "--pre-cropped", "--output", str(destination)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["is_valid"] and data["ita_angle"] is None and len(data["feature_vector"]) == 27
    assert (destination / "normalized.png").is_file()
    Image.fromarray(np.full((30, 30, 3), [170, 95, 100], dtype=np.uint8)).save(source)
    assert main(["image", str(source), "--body-part", "eye", "--pre-cropped", "--output", str(destination)]) == 2
    assert not json.loads(capsys.readouterr().out)["is_valid"]
    assert not (destination / "normalized.png").exists()


def test_cli_eda_crop_uses_white_background_mask(tmp_path, capsys):
    source = tmp_path / "segmented.png"
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[25:75, 25:75] = [170, 95, 100]
    Image.fromarray(image).save(source)
    assert main(["image", str(source), "--body-part", "eye", "--eda-segmented", "--output", str(tmp_path / "preview")]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["glare_fraction"] == 0 and data["metadata"]["roi_bbox"] == [25, 25, 75, 75]


def test_cli_bad_dataset_returns_nonzero_without_traceback(tmp_path, capsys):
    assert main(["prepare", "--eyes", str(tmp_path / "missing"), "--nails", str(tmp_path / "missing2"), "--output", str(tmp_path / "out")]) == 2
    assert "Error:" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
