from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soilgsd.constants import ID_COLUMN, LOG_WIDTHS, MAX_SCORE, TARGET_COLUMNS
from soilgsd.metric import score_frames, weighted_emd, weighted_emd_per_sample


def test_interval_weights_span_five_decades():
    # log10(200 / 0.002) = 5, which is what pins the metric's range at [0, 500].
    assert LOG_WIDTHS.sum() == pytest.approx(5.0)
    assert MAX_SCORE == pytest.approx(500.0)


def test_perfect_prediction_scores_zero():
    curve = np.linspace(0.0, 100.0, 11)[None, :]
    assert weighted_emd(curve, curve) == pytest.approx(0.0)


def test_worst_case_hits_the_upper_bound():
    # Everything passes the finest sieve versus nothing until the coarsest.
    all_fine = np.full((1, 11), 100.0)
    all_coarse = np.zeros((1, 11))
    all_coarse[0, -1] = 100.0
    assert weighted_emd(all_fine, all_coarse) == pytest.approx(MAX_SCORE)


def test_matches_the_explicit_reference_formula():
    rng = np.random.default_rng(0)
    truth = np.sort(rng.uniform(0, 100, size=(5, 11)), axis=1)
    truth[:, -1] = 100.0
    predicted = np.sort(rng.uniform(0, 100, size=(5, 11)), axis=1)
    predicted[:, -1] = 100.0

    supports = [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200]
    expected = []
    for row_true, row_pred in zip(truth, predicted):
        total = sum(
            abs(row_true[i] - row_pred[i])
            * (np.log10(supports[i + 1]) - np.log10(supports[i]))
            for i in range(10)
        )
        expected.append(total)

    assert weighted_emd_per_sample(truth, predicted) == pytest.approx(np.array(expected))


def test_final_support_is_ignored():
    # The 200 mm column is fixed at 100 and carries no interval of its own.
    a = np.linspace(0, 100, 11)[None, :]
    b = a.copy()
    b[0, -1] = 42.0
    assert weighted_emd(a, b) == pytest.approx(0.0)


def test_score_frames_aligns_on_sample_id():
    rng = np.random.default_rng(1)
    curves = np.sort(rng.uniform(0, 100, size=(4, 11)), axis=1)
    curves[:, -1] = 100.0
    truth = pd.DataFrame(curves, columns=list(TARGET_COLUMNS))
    truth.insert(0, ID_COLUMN, [f"s{i}" for i in range(4)])

    shuffled = truth.iloc[::-1].reset_index(drop=True)
    assert score_frames(truth, shuffled) == pytest.approx(0.0)


def test_score_frames_rejects_mismatched_ids():
    curves = np.tile(np.linspace(0, 100, 11), (2, 1))
    truth = pd.DataFrame(curves, columns=list(TARGET_COLUMNS))
    truth.insert(0, ID_COLUMN, ["a", "b"])
    other = truth.copy()
    other[ID_COLUMN] = ["a", "c"]
    with pytest.raises(ValueError, match="different sample_id sets"):
        score_frames(truth, other)


def test_shape_errors_are_explicit():
    with pytest.raises(ValueError, match="n_samples, 11"):
        weighted_emd(np.zeros((2, 10)), np.zeros((2, 10)))
