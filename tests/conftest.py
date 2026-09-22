from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """A small synthetic competition folder shared by the slower tests."""
    from soilgsd.synthetic import generate_dataset

    root = tmp_path_factory.mktemp("synthetic")
    return generate_dataset(root, n_train=12, n_test=4, photos_per_sample=2, size_px=128, seed=3)
