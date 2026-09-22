"""Photo features aligned to the physical millimetre axis of the target.

The target is indexed by grain diameter in millimetres, so the features should
be too.  Every photo is first resampled to a **fixed physical resolution**
(pixels per millimetre) using the camera scale from ``ppm_updated.csv``.  After
that step one pixel means the same physical length in every photo, whatever
camera took it, and a multi-scale decomposition becomes a granulometry: the
band energy at pyramid level ``l`` measures how much image contrast lives at a
physical scale of ``base_mm * 2**l``.

That is the signal the model needs.  Two honest limits are worth stating:

* The finest supports (clay at 0.002 mm, silt at 0.02 mm) are far below what a
  photograph resolves.  No texture feature can see them; they have to be
  inferred from colour, gloss and the coarse fraction, so expect the fine end
  of the curve to be the hardest part of the score.
* Band energy responds to grain *contrast* as well as grain *size*.  Colour and
  brightness statistics are included alongside so a model can separate the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ID_COLUMN

__all__ = ["FeatureConfig", "photo_features", "extract_photo_features", "aggregate_to_samples"]


@dataclass(frozen=True)
class FeatureConfig:
    """Settings for the physically-calibrated feature extractor."""

    #: Pixels per millimetre every photo is resampled to before analysis.
    target_ppm: float = 4.0
    #: Side of each analysis window, in pixels at ``target_ppm``. 256 px = 64 mm.
    crop_px: int = 256
    #: Analysis windows are taken on a ``crop_grid x crop_grid`` lattice.
    crop_grid: int = 3
    #: Number of pyramid levels; level ``l`` probes ``(1 / target_ppm) * 2**l`` mm.
    n_levels: int = 7
    #: Longest side a photo is allowed to reach after rescaling, to bound memory.
    max_side_px: int = 2048
    #: Fallback scale when a photo has no entry in the ppm table.
    default_ppm: float = 4.0
    quantiles: tuple[float, ...] = field(default=(0.05, 0.25, 0.5, 0.75, 0.95))

    def level_scales_mm(self) -> np.ndarray:
        """Physical scale, in millimetres, probed by each pyramid level."""
        base = 1.0 / self.target_ppm
        return base * 2.0 ** np.arange(self.n_levels, dtype=np.float64)


def _load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0


def _rescale_to_target_ppm(
    image: np.ndarray, ppm: float, config: FeatureConfig
) -> np.ndarray:
    """Resample so one pixel spans ``1 / target_ppm`` millimetres."""
    from PIL import Image

    factor = config.target_ppm / float(ppm)
    height, width = image.shape[:2]
    new_height = max(1, int(round(height * factor)))
    new_width = max(1, int(round(width * factor)))

    longest = max(new_height, new_width)
    if longest > config.max_side_px:
        shrink = config.max_side_px / longest
        new_height = max(1, int(round(new_height * shrink)))
        new_width = max(1, int(round(new_width * shrink)))

    if (new_height, new_width) == (height, width):
        return image
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
    # LANCZOS when shrinking keeps fine texture from aliasing into the wrong band.
    resample = Image.LANCZOS if factor < 1.0 else Image.BICUBIC
    pil = pil.resize((new_width, new_height), resample)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _effective_ppm(image_after: np.ndarray, image_before: np.ndarray, ppm: float) -> float:
    """The scale actually achieved, after any max-side clamp."""
    before = max(image_before.shape[0], image_before.shape[1])
    after = max(image_after.shape[0], image_after.shape[1])
    return float(ppm) * (after / before) if before else float(ppm)


def _crop_origins(size: int, crop: int, grid: int) -> list[int]:
    if size <= crop:
        return [0]
    if grid <= 1:
        return [(size - crop) // 2]
    step = (size - crop) / (grid - 1)
    return sorted({int(round(index * step)) for index in range(grid)})


def _downsample2(plane: np.ndarray) -> np.ndarray:
    """Average 2x2 blocks, trimming an odd final row or column."""
    height = plane.shape[0] - plane.shape[0] % 2
    width = plane.shape[1] - plane.shape[1] % 2
    if height < 2 or width < 2:
        return plane[:1, :1]
    trimmed = plane[:height, :width]
    return trimmed.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))


def _band_energies(gray: np.ndarray, n_levels: int) -> np.ndarray:
    """Contrast energy per octave, normalised to sum to one.

    The normalised profile is what carries size information; overall contrast is
    reported separately so a model is not forced to entangle the two.
    """
    energies = np.zeros(n_levels, dtype=np.float64)
    current = gray.astype(np.float64)
    for level in range(n_levels):
        coarse = _downsample2(current)
        if coarse.shape[0] < 1 or coarse.shape[1] < 1:
            break
        upsampled = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)
        height = min(current.shape[0], upsampled.shape[0])
        width = min(current.shape[1], upsampled.shape[1])
        detail = current[:height, :width] - upsampled[:height, :width]
        energies[level] = float(np.sqrt(np.mean(detail**2)))
        current = coarse
        if min(current.shape) < 2:
            break
    return energies


def _window_features(window: np.ndarray, config: FeatureConfig) -> dict[str, float]:
    red, green, blue = window[..., 0], window[..., 1], window[..., 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue

    out: dict[str, float] = {}
    for name, plane in (("r", red), ("g", green), ("b", blue), ("gray", gray)):
        out[f"{name}_mean"] = float(plane.mean())
        out[f"{name}_std"] = float(plane.std())
    for quantile in config.quantiles:
        out[f"gray_q{int(quantile * 100):02d}"] = float(np.quantile(gray, quantile))

    # Colour ratios are robust to overall exposure differences between cameras.
    total = red + green + blue + 1e-6
    out["chroma_rg"] = float(np.mean((red - green) / total))
    out["chroma_by"] = float(np.mean((blue - 0.5 * (red + green)) / total))
    out["saturation"] = float(np.mean(window.max(axis=-1) - window.min(axis=-1)))

    grad_y = np.diff(gray, axis=0)
    grad_x = np.diff(gray, axis=1)
    out["grad_rms"] = float(np.sqrt(np.mean(grad_y**2) + np.mean(grad_x**2)))

    energies = _band_energies(gray, config.n_levels)
    out["band_total"] = float(energies.sum())
    profile = energies / (energies.sum() + 1e-12)
    scales = config.level_scales_mm()
    for level, (value, scale) in enumerate(zip(profile, scales)):
        out[f"band{level}_mm{scale:g}"] = float(value)

    # Where the texture energy sits on the log-mm axis: a one-number summary of
    # apparent grain size, directly comparable with the target's own axis.
    log_scales = np.log10(scales)
    centroid = float((profile * log_scales).sum())
    out["band_centroid_log10mm"] = centroid
    out["band_spread_log10mm"] = float(
        np.sqrt(max((profile * (log_scales - centroid) ** 2).sum(), 0.0))
    )
    return out


def photo_features(
    path: str | Path,
    ppm: float | None,
    config: FeatureConfig | None = None,
) -> dict[str, float]:
    """Extract one feature vector for a single photograph."""
    config = config or FeatureConfig()
    scale = float(ppm) if ppm and np.isfinite(ppm) and ppm > 0 else config.default_ppm

    original = _load_rgb(path)
    image = _rescale_to_target_ppm(original, scale, config)
    achieved_ppm = _effective_ppm(image, original, scale)

    height, width = image.shape[:2]
    crop = min(config.crop_px, height, width)
    windows = [
        image[top : top + crop, left : left + crop]
        for top in _crop_origins(height, crop, config.crop_grid)
        for left in _crop_origins(width, crop, config.crop_grid)
    ]

    per_window = [_window_features(window, config) for window in windows]
    keys = list(per_window[0])
    stacked = np.array([[row[key] for key in keys] for row in per_window], dtype=np.float64)

    features = {key: float(value) for key, value in zip(keys, stacked.mean(axis=0))}
    # Spread across windows measures how heterogeneous the sample surface is,
    # which separates a well-graded soil from a uniform one.
    for key, value in zip(keys, stacked.std(axis=0)):
        features[f"{key}__wstd"] = float(value)

    features["meta_ppm"] = achieved_ppm
    features["meta_log_ppm"] = float(np.log10(max(achieved_ppm, 1e-6)))
    features["meta_field_mm"] = float(max(height, width) / max(achieved_ppm, 1e-6))
    features["meta_n_windows"] = float(len(windows))
    return features


def extract_photo_features(
    photo_index: pd.DataFrame,
    config: FeatureConfig | None = None,
    *,
    progress: bool = False,
) -> pd.DataFrame:
    """Extract features for every row of a photo index, keeping failures visible."""
    config = config or FeatureConfig()
    rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    iterator = photo_index.itertuples(index=False)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(photo_index), desc="features")
        except ImportError:
            pass

    for record in iterator:
        path = str(record.path)
        try:
            features = photo_features(path, getattr(record, "ppm", None), config)
        except Exception as error:  # noqa: BLE001 - one bad photo must not kill a run
            failures.append((path, f"{type(error).__name__}: {error}"))
            continue
        features[ID_COLUMN] = str(getattr(record, ID_COLUMN))
        features["photo_id"] = str(record.photo_id)
        features["path"] = path
        rows.append(features)

    if not rows:
        raise RuntimeError(
            f"no photo yielded features ({len(failures)} failed); first: {failures[:3]}"
        )

    frame = pd.DataFrame.from_records(rows)
    leading = [ID_COLUMN, "photo_id", "path"]
    frame = frame[leading + [c for c in frame.columns if c not in leading]]
    frame.attrs["failures"] = failures
    return frame.reset_index(drop=True)


def aggregate_to_samples(photo_feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-photo features to one row per sample.

    Photos of the same sample are repeat observations of one physical soil, so
    the mean pools them; the spread across photos is kept as an extra feature
    because disagreement between photos is itself informative.
    """
    numeric = [
        column
        for column in photo_feature_frame.columns
        if column not in {ID_COLUMN, "photo_id", "path"}
        and pd.api.types.is_numeric_dtype(photo_feature_frame[column])
    ]
    grouped = photo_feature_frame.groupby(ID_COLUMN, sort=True)[numeric]

    mean = grouped.mean()
    spread = grouped.std(ddof=0).fillna(0.0).add_suffix("__pstd")
    counts = photo_feature_frame.groupby(ID_COLUMN, sort=True).size().rename("meta_n_photos")

    out = pd.concat([mean, spread, counts], axis=1).reset_index()
    out[ID_COLUMN] = out[ID_COLUMN].astype(str)
    return out
