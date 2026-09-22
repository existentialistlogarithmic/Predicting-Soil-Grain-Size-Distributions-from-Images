from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgsd.constants import ID_COLUMN, TARGET_COLUMNS
from soilgsd.data import DataPaths, build_photo_index, load_labels, load_ppm, load_sample_submission
from soilgsd.features import FeatureConfig, aggregate_to_samples, extract_photo_features, photo_features


def test_discovery_finds_the_official_layout(synthetic_root):
    paths = DataPaths.discover(synthetic_root)
    assert paths.labels.name == "Training_labels_updated.csv"
    assert paths.train_photos.name == "Training-All_Photos_updated"
    assert paths.test_photos.name == "Test_All_Photos"
    assert paths.ppm is not None


def test_discovery_reports_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        DataPaths.discover(tmp_path)


def test_labels_load_as_valid_curves(synthetic_root):
    paths = DataPaths.discover(synthetic_root)
    labels = load_labels(paths.labels)
    assert list(labels.columns) == [ID_COLUMN, *TARGET_COLUMNS]
    values = labels[list(TARGET_COLUMNS)].to_numpy()
    assert (np.diff(values, axis=1) >= -1e-6).all()
    assert values[:, -1] == pytest.approx(100.0)


def test_sample_submission_gives_the_test_ids(synthetic_root):
    paths = DataPaths.discover(synthetic_root)
    template = load_sample_submission(paths.sample_submission)
    assert len(template) == 4
    assert template[ID_COLUMN].is_unique


def test_ppm_table_is_normalised(synthetic_root):
    paths = DataPaths.discover(synthetic_root)
    ppm = load_ppm(paths.ppm)
    assert set(ppm.columns) == {"camera_key", "label", "ppm", "reference_long_side"}
    assert (ppm["ppm"] > 0).all()
    assert (ppm["reference_long_side"] > 0).all()


def test_photo_index_maps_photos_onto_samples(synthetic_root):
    paths = DataPaths.discover(synthetic_root)
    labels = load_labels(paths.labels)
    index = build_photo_index(paths.train_photos, labels[ID_COLUMN], ppm=load_ppm(paths.ppm))
    assert len(index) == 2 * len(labels)
    assert set(index[ID_COLUMN]) == set(labels[ID_COLUMN])
    assert index["ppm"].notna().all()
    assert index.attrs["unmatched"] == []
    assert index.attrs["no_camera"] == []


def test_longest_sample_id_wins_the_match(tmp_path):
    from PIL import Image

    folder = tmp_path / "photos"
    folder.mkdir()
    for stem in ("S1_a", "S12_a"):
        Image.new("RGB", (32, 32), color=(10, 20, 30)).save(folder / f"{stem}.png")

    index = build_photo_index(folder, ["S1", "S12"])
    mapping = dict(zip(index["photo_id"], index[ID_COLUMN]))
    assert mapping == {"S1_a": "S1", "S12_a": "S12"}


def test_unmatched_photos_are_reported_not_dropped_silently(tmp_path):
    from PIL import Image

    folder = tmp_path / "photos"
    folder.mkdir()
    Image.new("RGB", (32, 32)).save(folder / "mystery.png")
    index = build_photo_index(folder, ["S1"])
    assert index.empty
    assert index.attrs["unmatched"] == ["mystery.png"]


def test_feature_scales_span_the_configured_millimetre_range():
    config = FeatureConfig(target_ppm=4.0, n_levels=7)
    scales = config.level_scales_mm()
    assert scales[0] == pytest.approx(0.25)
    assert scales[-1] == pytest.approx(16.0)


def test_features_are_finite_and_named(synthetic_root):
    paths = DataPaths.discover(synthetic_root)
    index = build_photo_index(paths.train_photos, load_labels(paths.labels)[ID_COLUMN])
    features = photo_features(index.loc[0, "path"], 4.0, FeatureConfig(crop_px=96, crop_grid=2))
    assert all(np.isfinite(value) for value in features.values())
    assert any(key.startswith("band0_mm") for key in features)
    assert "band_centroid_log10mm" in features


def test_scale_calibration_makes_features_camera_independent(tmp_path):
    """The same soil shot at two magnifications must look the same after rescaling."""
    from PIL import Image

    rng = np.random.default_rng(0)
    cells = 32
    coarse = rng.uniform(0.2, 0.8, size=(cells, cells))
    base = np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1)  # 256 px, blobs 8 px wide
    high = np.repeat(np.repeat(base, 2, axis=0), 2, axis=1)  # same soil, 2x the ppm

    low_path = tmp_path / "low.png"
    high_path = tmp_path / "high.png"
    Image.fromarray((base * 255).astype(np.uint8)).convert("RGB").save(low_path)
    Image.fromarray((high * 255).astype(np.uint8)).convert("RGB").save(high_path)

    config = FeatureConfig(target_ppm=4.0, crop_px=128, crop_grid=1, n_levels=6)
    low = photo_features(low_path, ppm=4.0, config=config)
    high = photo_features(high_path, ppm=8.0, config=config)

    # Without the ppm correction the texture centroid would shift by a full octave.
    assert low["band_centroid_log10mm"] == pytest.approx(
        high["band_centroid_log10mm"], abs=0.12
    )


def test_band_centroid_tracks_physical_grain_size(tmp_path):
    """Coarser soil must push texture energy to a coarser physical scale."""
    from PIL import Image

    rng = np.random.default_rng(1)
    config = FeatureConfig(target_ppm=4.0, crop_px=192, crop_grid=1, n_levels=7)

    centroids = []
    for block in (2, 16):  # blobs 0.5 mm and 4 mm wide at 4 px/mm
        cells = 256 // block
        coarse = rng.uniform(0.15, 0.85, size=(cells, cells))
        image = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)
        path = tmp_path / f"grain{block}.png"
        Image.fromarray((image * 255).astype(np.uint8)).convert("RGB").save(path)
        centroids.append(photo_features(path, ppm=4.0, config=config)["band_centroid_log10mm"])

    assert centroids[1] > centroids[0] + 0.3


def test_aggregation_pools_photos_into_one_row_per_sample(synthetic_root):
    paths = DataPaths.discover(synthetic_root)
    labels = load_labels(paths.labels)
    index = build_photo_index(paths.train_photos, labels[ID_COLUMN], ppm=load_ppm(paths.ppm))
    per_photo = extract_photo_features(index, FeatureConfig(crop_px=96, crop_grid=1))
    samples = aggregate_to_samples(per_photo)

    assert len(samples) == len(labels)
    assert samples[ID_COLUMN].is_unique
    assert (samples["meta_n_photos"] == 2).all()
    assert any(column.endswith("__pstd") for column in samples.columns)
    assert samples.drop(columns=[ID_COLUMN]).notna().all().all()
