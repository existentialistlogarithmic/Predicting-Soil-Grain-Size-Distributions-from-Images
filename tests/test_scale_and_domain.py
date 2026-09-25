"""Regression tests for the two traps the real archive sets.

Both were found by inspecting the competition data, and both are silent: the
pipeline runs happily while producing predictions that cannot score well.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soilgsd.constants import ID_COLUMN, TARGET_COLUMNS
from soilgsd.curves import project_valid
from soilgsd.data import build_photo_index, load_ppm, normalise_key, resolve_camera
from soilgsd.features import select_features
from soilgsd.models import NeighbourCurve, build_model


# --------------------------------------------------------------------------
# Trap 1: ppm is quoted at a reference resolution the file may not be at.
# --------------------------------------------------------------------------

def _write_ppm(tmp_path):
    path = tmp_path / "ppm.csv"
    pd.DataFrame(
        [
            {"phone": "Motorola Edge", "camera": "motorola edge 20",
             "width": 4000, "height": 1800, "ppm": 11.492},
            {"phone": "Motorola Edge 60 Fusion", "camera": "Motorola Edge 60 fusion",
             "width": 4096, "height": 2304, "ppm": 12.465},
            {"phone": "Samsung A52", "camera": "SM-A525F",
             "width": 9248, "height": 6936, "ppm": 26.330},
            {"phone": "iPhone 14", "camera": "iPhone 14",
             "width": 4032, "height": 3024, "ppm": 13.942},
        ]
    ).to_csv(path, index=False)
    return load_ppm(path)


def test_quoted_ppm_is_corrected_for_delivered_resolution(tmp_path):
    """A Samsung frame shipped at 1599 px is 4.55 ppm, not the quoted 26.33."""
    from PIL import Image

    ppm = _write_ppm(tmp_path)
    folder = tmp_path / "photos"
    folder.mkdir()
    Image.new("RGB", (1599, 1200), (120, 110, 100)).save(folder / "Samsung_A52_H030_01.jpg")
    Image.new("RGB", (1600, 720), (120, 110, 100)).save(folder / "Motorola_Edge_H030_02.jpg")

    index = build_photo_index(folder, ["H030"], ppm=ppm).set_index("photo_id")

    assert index.loc["Samsung_A52_H030_01", "ppm"] == pytest.approx(4.553, abs=0.01)
    assert index.loc["Motorola_Edge_H030_02", "ppm"] == pytest.approx(4.597, abs=0.01)
    # Taking the quoted number at face value would be off by a factor of 5.8.
    assert index.loc["Samsung_A52_H030_01", "ppm"] < 26.330 / 5


def test_native_resolution_photos_keep_their_quoted_scale(tmp_path):
    from PIL import Image

    ppm = _write_ppm(tmp_path)
    folder = tmp_path / "photos"
    folder.mkdir()
    Image.new("RGB", (4032, 3024), (120, 110, 100)).save(folder / "iPhone14_HPC_Audorfring (1).JPG")

    index = build_photo_index(folder, ["HPC_Audorfring"], ppm=ppm)
    assert index.loc[0, "ppm"] == pytest.approx(13.942, abs=0.01)
    assert index.loc[0, "resolution_scale"] == pytest.approx(1.0, abs=0.01)


def test_longest_camera_name_wins(tmp_path):
    """Motorola_Edge must not claim Motorola_Edge_60_fusion's photos."""
    ppm = _write_ppm(tmp_path)
    assert resolve_camera("Motorola_Edge_H030_01", ppm)["label"] == "Motorola Edge"
    plain = resolve_camera("Motorola_Edge_60_fusion_H374_01", ppm)
    assert plain["label"] == "Motorola Edge 60 Fusion"
    assert plain["ppm"] == pytest.approx(12.465)


# --------------------------------------------------------------------------
# Trap 2: inconsistent spellings between the photos and sample_submission.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "photo_stem, sample_id",
    [
        # The real pair: a mangled umlaut and a comma against an underscore.
        ("iPhone14_HPC_M#U00fcnster_BS6_9,0-10m (3)", "HPC_Muenster_BS6_9_0-10m"),
        ("iPhone16_HPC_Kleinkummerfeld 18-3 (2)", "HPC_Kleinkummerfeld 18-3"),
        ("iPhone_16_HPC_Airbus BS10-4bis7 (4)", "HPC_Airbus BS10-4bis7"),
        ("Samsung_A52_H030_01", "H030"),
    ],
)
def test_photo_filenames_reach_their_sample(tmp_path, photo_stem, sample_id):
    from PIL import Image

    folder = tmp_path / "photos"
    folder.mkdir()
    Image.new("RGB", (64, 48)).save(folder / f"{photo_stem}.jpg")

    index = build_photo_index(folder, [sample_id])
    assert index.attrs["unmatched"] == []
    assert index.loc[0, ID_COLUMN] == sample_id


def test_normalise_key_folds_the_spellings_that_differ():
    assert normalise_key("HPC_M#U00fcnster_BS6_9,0-10m") == normalise_key("HPC_Muenster_BS6_9_0-10m")
    assert normalise_key("iPhone_16") == normalise_key("iPhone16")
    assert normalise_key("Motorola Edge") == "motorolaedge"


# --------------------------------------------------------------------------
# Feature selection: keep the camera out of the model.
# --------------------------------------------------------------------------

COLUMNS = [
    "band0_mm0.25", "band3_mm2", "band0_mm0.25__wstd", "band3_mm2__pstd",
    "grad_rms", "grad_rms__pstd", "gray_mean", "r_std", "chroma_rg",
    "saturation", "meta_ppm", "meta_log_ppm", "meta_n_photos",
]


def test_texture_set_excludes_camera_identifying_features():
    chosen = select_features(COLUMNS, "texture")
    assert not [c for c in chosen if c.startswith("meta_")], "meta_* identifies the camera"
    assert not [c for c in chosen if "__pstd" in c], "__pstd mixes cameras in train only"
    assert not [c for c in chosen if c.startswith(("gray", "r_", "chroma", "satur"))]
    assert "band0_mm0.25" in chosen and "band0_mm0.25__wstd" in chosen


def test_texture_full_keeps_pstd_but_still_drops_meta():
    chosen = select_features(COLUMNS, "texture_full")
    assert "band3_mm2__pstd" in chosen
    assert not [c for c in chosen if c.startswith("meta_")]


def test_all_set_still_drops_meta():
    assert not [c for c in select_features(COLUMNS, "all") if c.startswith("meta_")]


def test_unknown_feature_set_is_rejected():
    with pytest.raises(ValueError, match="unknown feature_set"):
        select_features(COLUMNS, "nonsense")


# --------------------------------------------------------------------------
# Per-domain standardisation.
# --------------------------------------------------------------------------

def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.uniform(-1.0, 1.0, size=n)
    grid = np.linspace(-1.5, 1.5, 11)
    curves = project_valid(
        100.0 / (1.0 + np.exp(-(grid[None, :] - signal[:, None] * 1.5) * 2.2))
    )
    features = pd.DataFrame({"a": signal, "b": signal * 0.5 + rng.normal(scale=0.1, size=n)})
    return features, pd.DataFrame(curves, columns=list(TARGET_COLUMNS))


def test_per_domain_standardisation_absorbs_a_domain_shift():
    """A batch offset and rescaled wholesale predicts the same as the original."""
    features, curves = _toy()
    model = NeighbourCurve(n_neighbours=7, per_domain=True).fit(features, curves)

    shifted = features * 3.0 + 25.0
    assert model.predict(shifted) == pytest.approx(model.predict(features))


def test_without_per_domain_a_shifted_batch_collapses():
    features, curves = _toy()
    model = NeighbourCurve(n_neighbours=7, per_domain=False).fit(features, curves)

    shifted = features * 3.0 + 25.0
    predictions = model.predict(shifted)
    # Every row lands on the same few neighbours, so variety disappears.
    assert np.unique(predictions, axis=0).shape[0] < len(features) / 4


def test_small_batches_fall_back_to_training_statistics():
    features, curves = _toy()
    model = NeighbourCurve(n_neighbours=5, per_domain=True, min_domain_rows=5).fit(features, curves)

    one = features.iloc[[0]]
    reference = NeighbourCurve(n_neighbours=5, per_domain=False).fit(features, curves)
    assert model.predict(one) == pytest.approx(reference.predict(one))


def test_default_neighbour_model_uses_the_tuned_settings():
    """Per-domain scaling measured better on every offline diagnostic and cost
    17 EMD on the leaderboard, so the default must stay off."""
    model = build_model("knn")
    assert isinstance(model, NeighbourCurve)
    assert model.n_neighbours == 7
    assert model.per_domain is False


# --------------------------------------------------------------------------
# Curves read from the photographs.
# --------------------------------------------------------------------------

def test_visual_readings_cover_every_test_sample():
    from soilgsd.visual import load_readings

    readings, weight = load_readings()
    assert len(readings) == 10
    assert 0.0 <= weight <= 1.0
    for name, entry in readings.items():
        assert entry["d50_mm"] > 0, name
        assert 0.3 <= entry["sigma"] <= 1.5, name
        assert entry.get("note"), f"{name} needs a note saying what was seen"


def test_readings_build_valid_curves_ordered_by_grain_size():
    from soilgsd.curves import is_valid
    from soilgsd.models import _log_d50
    from soilgsd.visual import curves_from_readings, load_readings

    readings, _ = load_readings()
    names = list(readings)
    curves = curves_from_readings(readings, names)
    assert is_valid(curves).all()

    recovered = 10 ** _log_d50(curves)
    stated = np.array([readings[n]["d50_mm"] for n in names])
    # The curve really does pass 50% at the diameter that was read off the bar.
    assert np.allclose(recovered, stated, rtol=0.12)


def test_reading_reaches_past_the_training_range():
    """The point of reading by eye: the coarsest soils are outside training."""
    from soilgsd.visual import load_readings

    readings, _ = load_readings()
    assert readings["HPC_Muenster_BS6_9_0-10m"]["d50_mm"] > 6.13


def test_blend_of_two_valid_curves_is_valid():
    from soilgsd.curves import is_valid, project_valid
    from soilgsd.visual import blend_with_model

    rng = np.random.default_rng(0)
    a = project_valid(rng.uniform(0, 100, size=(6, 11)))
    b = project_valid(rng.uniform(0, 100, size=(6, 11)))
    assert is_valid(blend_with_model(a, b, 0.5)).all()
    assert blend_with_model(a, b, 1.0) == pytest.approx(a)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        blend_with_model(a, b, 1.5)


def test_warping_lands_on_the_reading():
    """Shape comes from the model; position and width come from the photo."""
    from soilgsd.curves import is_valid
    from soilgsd.models import _log_d50
    from soilgsd.visual import curve_spread, warp_to_reading

    rng = np.random.default_rng(0)
    shapes = project_valid(np.sort(rng.uniform(0, 100, size=(6, 11)), axis=1))
    targets = np.array([0.4, 0.8, 2.0, 6.0, 18.0, 30.0])
    widths = np.full(6, 1.3)

    warped = warp_to_reading(shapes, targets, widths)
    assert is_valid(warped).all()
    assert 10 ** _log_d50(warped) == pytest.approx(targets, rel=0.10)
    assert np.array([curve_spread(c) for c in warped]) == pytest.approx(widths, abs=0.25)


def test_warping_reaches_past_the_training_range():
    """The whole point: a pooled training curve cannot get to 30 mm, a warp can."""
    from soilgsd.models import _log_d50
    from soilgsd.visual import warp_to_reading

    pooled = project_valid(
        np.array([[2.0, 5, 12, 25, 45, 62, 78, 90, 96, 99, 100.0]])
    )
    assert 10 ** _log_d50(pooled)[0] < 1.0
    warped = warp_to_reading(pooled, [30.0])
    assert 10 ** _log_d50(warped)[0] == pytest.approx(30.0, rel=0.10)


def test_warp_rejects_mismatched_inputs():
    from soilgsd.visual import warp_to_reading

    shapes = project_valid(np.sort(np.random.default_rng(1).uniform(0, 100, size=(3, 11)), axis=1))
    with pytest.raises(ValueError, match="3 curves and 2 targets"):
        warp_to_reading(shapes, [1.0, 2.0])
    with pytest.raises(ValueError, match="spread_decades"):
        warp_to_reading(shapes, [1.0, 2.0, 3.0], [1.0])


def test_template_curves_land_on_the_reading_and_stay_valid():
    from soilgsd.curves import is_valid
    from soilgsd.models import _log_d50
    from soilgsd.visual import curves_from_template, load_readings

    readings, _ = load_readings()
    names = list(readings)
    # stand-in training curves spanning fine to coarse
    grid = np.linspace(-2.7, 2.3, 11)
    training = project_valid(
        np.array([100 / (1 + np.exp(-(grid - c) * 2.0)) for c in (-2.0, -0.5, 0.5, 1.4)])
    )
    curves = curves_from_template(readings, names, training)

    assert is_valid(curves).all()
    stated = np.array([readings[n]["d50_mm"] for n in names])
    assert 10 ** _log_d50(curves) == pytest.approx(stated, rel=0.12)


def test_template_beats_a_lognormal_on_the_real_curves():
    """Given only a d50 and a spread, an empirical template reproduces a real
    soil curve better than a lognormal does."""
    from scipy.stats import norm

    from soilgsd.constants import LOG_SUPPORTS, TARGET_COLUMNS
    from soilgsd.data import load_labels
    from soilgsd.metric import weighted_emd
    from soilgsd.models import _log_d50
    from soilgsd.visual import curve_spread, warp_to_reading

    try:
        truth = load_labels("data/raw/Training_labels_updated.csv")[list(TARGET_COLUMNS)].to_numpy()
    except FileNotFoundError:
        pytest.skip("competition data not present")

    centres = _log_d50(truth)
    spreads = np.array([curve_spread(c) for c in truth])

    lognormal, template = [], []
    for i, (centre, spread) in enumerate(zip(centres, spreads)):
        lognormal.append(project_valid(100 * norm.cdf((LOG_SUPPORTS - centre) / max(spread / 2, 0.05))))
        others = [j for j in range(len(truth)) if j != i]
        pick = others[int(np.argmin(np.abs(centres[others] - centre)))]
        template.append(warp_to_reading(truth[pick][None, :], [10**centre], [spread])[0])

    assert weighted_emd(truth, np.array(template)) < weighted_emd(truth, np.array(lognormal))
