"""Conversions and projections for cumulative grain-size curves.

A *valid* curve is a vector of eleven percentages that is non-decreasing, lies
in ``[0, 100]``, and ends at exactly 100.  Kaggle rejects anything else, so
every prediction leaves the library through :func:`project_valid`.

The natural unconstrained parameterisation is the vector of **bin masses**: the
percentage of the sample falling in each of the eleven size bins (the first bin
being everything finer than 0.002 mm, the last everything between 63 and
200 mm).  Masses are non-negative and sum to 100, so a model that predicts a
point on the simplex and takes a cumulative sum satisfies all three constraints
by construction rather than by post-hoc repair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import N_SUPPORTS, TARGET_COLUMNS, VALUE_MAX, VALUE_MIN

__all__ = [
    "project_valid",
    "curves_to_masses",
    "masses_to_curves",
    "median_curve",
    "is_valid",
]


def _as_2d(values: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, bool]:
    if isinstance(values, pd.DataFrame):
        values = values.reindex(columns=list(TARGET_COLUMNS)).to_numpy(dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    was_1d = array.ndim == 1
    if was_1d:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != N_SUPPORTS:
        raise ValueError(
            f"expected an array of shape (n_samples, {N_SUPPORTS}), got {array.shape}"
        )
    return array, was_1d


def _isotonic_row(row: np.ndarray) -> np.ndarray:
    """Least-squares isotonic (non-decreasing) fit via pool-adjacent-violators."""
    values: list[float] = []
    weights: list[float] = []
    for value in row:
        values.append(float(value))
        weights.append(1.0)
        while len(values) > 1 and values[-2] > values[-1]:
            weight = weights[-1] + weights[-2]
            pooled = (values[-1] * weights[-1] + values[-2] * weights[-2]) / weight
            values[-2:] = [pooled]
            weights[-2:] = [weight]
    out = np.empty(row.shape[0], dtype=np.float64)
    position = 0
    for value, weight in zip(values, weights):
        count = int(round(weight))
        out[position : position + count] = value
        position += count
    return out


def project_valid(
    values: np.ndarray | pd.DataFrame,
    *,
    method: str = "isotonic",
) -> np.ndarray:
    """Project arbitrary predictions onto the set of valid cumulative curves.

    ``method='isotonic'`` uses the least-squares monotone fit, which spreads a
    violation across the offending run.  ``method='cummax'`` instead drags each
    dip up to the running maximum, which never lowers a value.  Both then clip
    to ``[0, 100]`` and pin the coarsest support to 100.

    ``NaN`` entries are treated as missing and filled by carrying the previous
    valid value forward (and 0 at the fine end), so a partially failed model
    still yields a submittable row.
    """
    array, was_1d = _as_2d(values)
    array = array.copy()

    if np.isnan(array).any():
        array[:, 0] = np.where(np.isnan(array[:, 0]), VALUE_MIN, array[:, 0])
        for column in range(1, N_SUPPORTS):
            previous = array[:, column - 1]
            array[:, column] = np.where(np.isnan(array[:, column]), previous, array[:, column])

    if method == "isotonic":
        array = np.vstack([_isotonic_row(row) for row in array])
    elif method == "cummax":
        array = np.maximum.accumulate(array, axis=1)
    else:
        raise ValueError(f"unknown projection method: {method!r}")

    array = np.clip(array, VALUE_MIN, VALUE_MAX)
    array[:, -1] = VALUE_MAX
    # Clipping the last column up to 100 can only have widened the final gap, so
    # monotonicity still holds; re-assert it cheaply for numerical safety.
    array = np.maximum.accumulate(array, axis=1)
    return array[0] if was_1d else array


def curves_to_masses(values: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Convert cumulative curves to per-bin masses summing to 100."""
    array, was_1d = _as_2d(values)
    masses = np.empty_like(array)
    masses[:, 0] = array[:, 0]
    masses[:, 1:] = np.diff(array, axis=1)
    return masses[0] if was_1d else masses


def masses_to_curves(masses: np.ndarray) -> np.ndarray:
    """Convert per-bin masses back to cumulative curves, then project to valid."""
    array = np.asarray(masses, dtype=np.float64)
    was_1d = array.ndim == 1
    if was_1d:
        array = array[None, :]
    array = np.clip(array, 0.0, None)
    totals = array.sum(axis=1, keepdims=True)
    # A degenerate all-zero row would divide by zero; put all its mass in the
    # coarsest bin, which is the least-committal valid curve.
    degenerate = (totals <= 0).ravel()
    if degenerate.any():
        array[degenerate] = 0.0
        array[degenerate, -1] = 1.0
        totals[degenerate] = 1.0
    curves = np.cumsum(array / totals * VALUE_MAX, axis=1)
    curves[:, -1] = VALUE_MAX
    return curves[0] if was_1d else curves


def median_curve(values: np.ndarray | pd.DataFrame) -> np.ndarray:
    """The pointwise median curve of a set of curves.

    Taking the median support by support preserves monotonicity, so the result
    is itself a valid curve.  Under an absolute-error metric this is the
    score-optimal constant prediction.
    """
    array, _ = _as_2d(values)
    return project_valid(np.median(array, axis=0))


def is_valid(values: np.ndarray | pd.DataFrame, *, tolerance: float = 1e-8) -> np.ndarray:
    """Boolean mask of which rows Kaggle would accept."""
    array, was_1d = _as_2d(values)
    finite = np.isfinite(array).all(axis=1)
    in_range = ((array >= VALUE_MIN - tolerance) & (array <= VALUE_MAX + tolerance)).all(axis=1)
    monotone = (np.diff(array, axis=1) >= -tolerance).all(axis=1)
    terminal = np.abs(array[:, -1] - VALUE_MAX) <= 1e-6
    mask = finite & in_range & monotone & terminal
    return mask[0] if was_1d else mask
