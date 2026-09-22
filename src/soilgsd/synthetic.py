"""Generate a fake competition folder with the official file layout.

The real photos and labels are under competition terms and need a Kaggle
account, but the pipeline should be testable and reviewable without them.  This
module renders soils whose visible texture scale genuinely tracks their grain
size distribution, so a model that works here is at least wired up correctly.

It is a rig, not a simulator: scores on synthetic data say nothing about the
leaderboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, SUBMISSION_COLUMNS, SUPPORTS_MM, TARGET_COLUMNS

__all__ = ["generate_dataset"]


def _lognormal_cdf(supports_mm: np.ndarray, median_mm: float, sigma_log10: float) -> np.ndarray:
    """Percent passing for a lognormal grain size distribution."""
    from math import erf, sqrt

    z = (np.log10(supports_mm) - np.log10(median_mm)) / (sigma_log10 * sqrt(2.0))
    return 50.0 * (1.0 + np.array([erf(float(value)) for value in z]))


def _render_soil(
    rng: np.random.Generator,
    median_mm: float,
    sigma_log10: float,
    ppm: float,
    size_px: int,
    tint: np.ndarray,
) -> np.ndarray:
    """Render a texture whose blob scale matches the grain distribution."""
    field = np.zeros((size_px, size_px), dtype=np.float64)
    # Superpose octaves centred on the distribution: each octave is a coarse
    # random grid stretched up, so its blobs are about one grain wide.
    for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
        grain_mm = median_mm * 10.0 ** (offset * sigma_log10)
        grain_px = max(2.0, grain_mm * ppm)
        cells = max(2, int(round(size_px / grain_px)))
        coarse = rng.normal(size=(cells, cells))
        repeat = int(np.ceil(size_px / cells))
        stretched = np.repeat(np.repeat(coarse, repeat, axis=0), repeat, axis=1)
        weight = float(np.exp(-0.5 * (offset / 0.8) ** 2))
        field += weight * stretched[:size_px, :size_px]

    field -= field.mean()
    spread = field.std()
    if spread > 0:
        field /= spread
    gray = np.clip(0.5 + 0.18 * field, 0.02, 0.98)
    image = gray[..., None] * tint[None, None, :]
    image += rng.normal(scale=0.01, size=image.shape)
    return np.clip(image, 0.0, 1.0)


def generate_dataset(
    root: str | Path,
    *,
    n_train: int = 24,
    n_test: int = 8,
    photos_per_sample: int = 2,
    size_px: int = 192,
    seed: int = 0,
) -> Path:
    """Write a synthetic competition folder under ``root`` and return the path.

    The layout mirrors the official archive: ``Training_labels_updated.csv``,
    ``sample_submission.csv``, ``ppm_updated.csv``, ``Training-All_Photos_updated/``
    and ``Test_All_Photos/``.
    """
    from PIL import Image

    root = Path(root)
    train_dir = root / "Training-All_Photos_updated"
    test_dir = root / "Test_All_Photos"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    supports = np.asarray(SUPPORTS_MM, dtype=np.float64)
    # Two camera scales, so the ppm correction has something to actually fix.
    cameras = {"camA": 6.0, "camB": 2.5}

    def build(prefix: str, count: int, directory: Path) -> tuple[list[str], list[np.ndarray], list[dict]]:
        ids: list[str] = []
        curves: list[np.ndarray] = []
        ppm_rows: list[dict] = []
        for index in range(count):
            sample_id = f"{prefix}{index:03d}"
            median_mm = float(10.0 ** rng.uniform(-1.3, 1.3))
            sigma = float(rng.uniform(0.35, 0.95))
            tint = np.clip(rng.normal(loc=[0.78, 0.70, 0.60], scale=0.09), 0.25, 1.0)
            for photo in range(photos_per_sample):
                camera = list(cameras)[(index + photo) % len(cameras)]
                ppm = cameras[camera]
                image = _render_soil(rng, median_mm, sigma, ppm, size_px, tint)
                photo_id = f"{sample_id}_p{photo}"
                Image.fromarray((image * 255).astype(np.uint8)).save(
                    directory / f"{photo_id}.jpg", quality=92
                )
                ppm_rows.append({"photo": f"{photo_id}.jpg", "camera": camera, "ppm": ppm})
            ids.append(sample_id)
            curves.append(_lognormal_cdf(supports, median_mm, sigma))
        return ids, curves, ppm_rows

    train_ids, train_curves, train_ppm = build("TRAIN", n_train, train_dir)
    test_ids, _, test_ppm = build("TEST", n_test, test_dir)

    labels = pd.DataFrame(np.vstack(train_curves), columns=list(TARGET_COLUMNS))
    labels[list(TARGET_COLUMNS)] = labels[list(TARGET_COLUMNS)].round(3)
    labels[TARGET_COLUMNS[-1]] = 100.0
    labels.insert(0, ID_COLUMN, train_ids)
    labels.to_csv(root / "Training_labels_updated.csv", index=False)

    template = pd.DataFrame(
        np.tile(np.linspace(0.0, 100.0, len(TARGET_COLUMNS)), (len(test_ids), 1)),
        columns=list(TARGET_COLUMNS),
    )
    template.insert(0, ID_COLUMN, test_ids)
    template.loc[:, list(SUBMISSION_COLUMNS)].to_csv(root / "sample_submission.csv", index=False)

    pd.DataFrame(train_ppm + test_ppm).to_csv(root / "ppm_updated.csv", index=False)
    return root
