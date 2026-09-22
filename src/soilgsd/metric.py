"""The competition metric: log-diameter-weighted Earth Mover's Distance.

For a single sample with true curve ``F`` and predicted curve ``Fhat`` sampled at
the eleven support diameters ``x``, the score is the area between the two
cumulative curves measured on a log10 diameter axis::

    EMD = sum_{i=0..9} |F_i - Fhat_i| * (log10(x_{i+1}) - log10(x_i))

The competition score is the mean of that over all scored samples.  Lower is
better; 0 is a perfect prediction and 500 is the worst possible value (100
percentage points of error across the full five decades of the axis).

Two consequences are worth keeping in mind while modelling:

* The ten interval widths are all within 0.4% of 0.5, so the metric is very
  nearly ``5 x MAE`` over the first ten supports.  The eleventh support is
  fixed at 100 and never enters the sum.
* Because it is an absolute-error metric, the score-optimal point prediction is
  the conditional **median** of each support, not the mean.  Train with an L1
  objective, and prefer medians when aggregating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, LOG_WIDTHS, N_SUPPORTS, TARGET_COLUMNS

__all__ = [
    "weighted_emd",
    "weighted_emd_per_sample",
    "score_frames",
]


def _as_curve_array(values: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        values = values.reindex(columns=list(TARGET_COLUMNS)).to_numpy(dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != N_SUPPORTS:
        raise ValueError(
            f"expected an array of shape (n_samples, {N_SUPPORTS}), got {array.shape}"
        )
    return array


def weighted_emd_per_sample(
    y_true: np.ndarray | pd.DataFrame,
    y_pred: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Return the per-sample weighted EMD, shape ``(n_samples,)``."""
    true = _as_curve_array(y_true)
    pred = _as_curve_array(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    # The final support is fixed at 100 for every valid curve and carries no
    # interval of its own, so only the first ten enter the sum.
    absolute_error = np.abs(true[:, :-1] - pred[:, :-1])
    return absolute_error @ LOG_WIDTHS


def weighted_emd(
    y_true: np.ndarray | pd.DataFrame,
    y_pred: np.ndarray | pd.DataFrame,
) -> float:
    """Return the competition score: the mean weighted EMD over all samples."""
    per_sample = weighted_emd_per_sample(y_true, y_pred)
    if per_sample.size == 0:
        raise ValueError("cannot score an empty set of samples")
    return float(per_sample.mean())


def score_frames(truth: pd.DataFrame, submission: pd.DataFrame) -> float:
    """Score a submission frame against a truth frame, aligning on ``sample_id``.

    Both frames must carry :data:`~soilgsd.constants.ID_COLUMN` and the eleven
    target columns, and must cover exactly the same set of sample ids.
    """
    for name, frame in (("truth", truth), ("submission", submission)):
        missing = {ID_COLUMN, *TARGET_COLUMNS} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} frame is missing columns: {sorted(missing)}")

    truth_indexed = truth.set_index(ID_COLUMN)
    submission_indexed = submission.set_index(ID_COLUMN)
    if truth_indexed.index.has_duplicates or submission_indexed.index.has_duplicates:
        raise ValueError("sample_id must be unique in both frames")
    if set(truth_indexed.index) != set(submission_indexed.index):
        raise ValueError("truth and submission cover different sample_id sets")

    submission_indexed = submission_indexed.reindex(truth_indexed.index)
    return weighted_emd(
        truth_indexed[list(TARGET_COLUMNS)],
        submission_indexed[list(TARGET_COLUMNS)],
    )
