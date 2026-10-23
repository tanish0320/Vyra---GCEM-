import importlib.util
from pathlib import Path
import sys

import cv2
import pytest


@pytest.fixture(scope="session", autouse=True)
def single_threaded_opencv():
    cv2.setNumThreads(1)


@pytest.fixture(scope="session")
def person2_reference():
    path = Path(__file__).parent / "fixtures" / "person2_reference_adapter.py"
    spec = importlib.util.spec_from_file_location("person2_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

