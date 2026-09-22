from __future__ import annotations

import numpy as np
import pytest

from soilgsd.curves import (
    curves_to_masses,
    is_valid,
    masses_to_curves,
    median_curve,
    project_valid,
)


def test_projection_makes_arbitrary_rows_submittable():
    rng = np.random.default_rng(0)
    raw = rng.uniform(-50, 150, size=(50, 11))
    assert is_valid(project_valid(raw)).all()
    assert is_valid(project_valid(raw, method="cummax")).all()


def test_projection_leaves_valid_curves_alone():
    curve = np.array([[0.0, 1.0, 5.0, 12.0, 30.0, 55.0, 75.0, 90.0, 96.0, 99.0, 100.0]])
    assert project_valid(curve) == pytest.approx(curve)


def test_cummax_never_lowers_a_value():
    raw = np.array([[10.0, 4.0, 30.0, 25.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0]])
    projected = project_valid(raw, method="cummax")
    assert (projected >= np.clip(raw, 0, 100) - 1e-9).all()


def test_nan_entries_are_carried_forward():
    raw = np.array([[np.nan, 10.0, np.nan, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, np.nan]])
    projected = project_valid(raw)
    assert is_valid(projected).all()
    assert projected[0, 0] == pytest.approx(0.0)
    assert projected[0, 2] == pytest.approx(10.0)


def test_mass_roundtrip_is_lossless():
    rng = np.random.default_rng(2)
    curves = project_valid(rng.uniform(0, 100, size=(20, 11)))
    assert masses_to_curves(curves_to_masses(curves)) == pytest.approx(curves)


def test_masses_are_non_negative_and_sum_to_one_hundred():
    rng = np.random.default_rng(3)
    curves = project_valid(rng.uniform(0, 100, size=(10, 11)))
    masses = curves_to_masses(curves)
    assert (masses >= -1e-9).all()
    assert masses.sum(axis=1) == pytest.approx(np.full(10, 100.0))


def test_degenerate_masses_still_yield_a_valid_curve():
    curves = masses_to_curves(np.zeros((3, 11)))
    assert is_valid(curves).all()


def test_pointwise_median_preserves_validity():
    rng = np.random.default_rng(4)
    curves = project_valid(rng.uniform(0, 100, size=(31, 11)))
    assert is_valid(median_curve(curves))


def test_median_beats_mean_under_the_metric():
    # The metric is absolute error, so the median is the optimal constant.
    from soilgsd.metric import weighted_emd

    rng = np.random.default_rng(5)
    curves = project_valid(np.sort(rng.lognormal(3.0, 1.0, size=(200, 11)), axis=1))
    median = np.tile(median_curve(curves), (len(curves), 1))
    mean = np.tile(project_valid(curves.mean(axis=0)), (len(curves), 1))
    assert weighted_emd(curves, median) <= weighted_emd(curves, mean)
