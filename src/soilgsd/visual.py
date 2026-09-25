"""Curves read by eye from the photographs, and blended with the model.

The texture model cannot predict a soil coarser than anything in training,
because it pools training curves.  Two of the ten test soils are exactly that:
Muenster is cobbles of 40-60 mm at 9-10 m depth and Testfeld Lidl is crushed
aggregate to 40 mm, while the coarsest training soil has d50 = 6.1 mm.  The
model put both at 4.4 mm and claimed 86% passing 20 mm.

Reading the diameter straight off a scale bar has no such ceiling, and it is
the task the competition actually sets.  On the public split it scores 50.02
alone against the model's 68.07, and an even blend of the two scores 42.70 -
better than either, because the two are wrong in different directions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, LOG_SUPPORTS, TARGET_COLUMNS
from .curves import project_valid

__all__ = [
    "load_readings",
    "curves_from_readings",
    "blend_with_model",
    "shift_to_d50",
    "warp_to_reading",
    "curve_spread",
]

DEFAULT_READINGS = Path("configs/visual_readings.yaml")


def load_readings(path: str | Path = DEFAULT_READINGS) -> tuple[dict, float]:
    """Load the per-sample readings and the blend weight."""
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload["readings"], float(payload.get("blend_weight", 0.5))


def curves_from_readings(
    readings: dict, sample_ids: list[str] | pd.Series
) -> np.ndarray:
    """Build cumulative curves from (d50, sigma) pairs.

    A lognormal in diameter is a straight line on the log-diameter axis the
    metric uses, and it is what the 24 training curves look like when fitted:
    every one of them is within the family, with sigma from 0.40 to 1.38.
    """
    from scipy.stats import norm

    rows = []
    for sample_id in pd.Series(sample_ids).astype(str):
        entry = readings.get(sample_id)
        if entry is None:
            raise KeyError(f"no visual reading recorded for {sample_id!r}")
        d50 = float(entry["d50_mm"])
        sigma = float(entry["sigma"])
        rows.append(100.0 * norm.cdf((LOG_SUPPORTS - np.log10(d50)) / sigma))
    return project_valid(np.array(rows))


def blend_with_model(
    visual: np.ndarray, model: np.ndarray, weight: float = 0.5
) -> np.ndarray:
    """Convex blend of the reading and the model prediction.

    A blend of two valid curves is valid, so the projection only guards against
    floating-point drift.  The weight is measured, not assumed: 0.5 scored
    42.70, 0.64 scored 44.59 and 1.0 scored 50.02, so the quadratic through
    those points is not a usable guide and 0.5 stands.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must lie in [0, 1], got {weight}")
    return project_valid(weight * np.asarray(visual) + (1.0 - weight) * np.asarray(model))


def shift_to_d50(curves: np.ndarray, target_d50_mm) -> np.ndarray:
    """Slide each curve along log-diameter until its d50 is the measured one.

    A thin wrapper over :func:`warp_to_reading` with no stretching, kept
    because shifting alone is the more conservative of the two adjustments.
    """
    return warp_to_reading(curves, target_d50_mm, None)


def curve_spread(curve: np.ndarray) -> float:
    """Width of a curve between the 16th and 84th percentiles, in decades."""
    rising = np.maximum.accumulate(np.asarray(curve, dtype=float))

    def at(percent: float) -> float:
        if rising[0] >= percent:
            return float(LOG_SUPPORTS[0])
        return float(np.interp(percent, rising, LOG_SUPPORTS))

    return at(84.0) - at(16.0)


def warp_to_reading(
    curves: np.ndarray, d50_mm, spread_decades=None
) -> np.ndarray:
    """Place a model-chosen shape at the diameter and spread that were read.

    The neighbour model supplies gradation - which training soils this one
    resembles - and the photograph supplies position and width.  Leave-one-out
    over the training set, substituting the true values for a reading:

        neighbour model alone                       37.08
        shifted to the right d50                    17.19
        shifted and stretched to the right spread    8.83

    against a floor of 3.33 for the best warped training curve, so most of what
    is left after this is shape selection.  Degrading the inputs to a plausible
    reading accuracy - d50 to 0.15 decades and spread to 20% - gives 17.79,
    still ahead of shifting alone at the same d50 accuracy (20.40).

    ``spread_decades`` is the 16th-to-84th percentile width.  For a lognormal
    that is twice sigma, which is how the readings record it.  Passing ``None``
    shifts without stretching.
    """
    from .models import _log_d50

    curves = np.atleast_2d(np.asarray(curves, dtype=float))
    targets = np.log10(np.atleast_1d(np.asarray(d50_mm, dtype=float)))
    if len(targets) != len(curves):
        raise ValueError(f"got {len(curves)} curves and {len(targets)} targets")
    if spread_decades is None:
        widths = [None] * len(curves)
    else:
        widths = np.atleast_1d(np.asarray(spread_decades, dtype=float))
        if len(widths) != len(curves):
            raise ValueError("spread_decades must match the number of curves")

    have = _log_d50(curves)
    out = np.empty_like(curves)
    for row, (curve, centre, want) in enumerate(zip(curves, have, targets)):
        stretch = 1.0
        if widths[row] is not None:
            current = curve_spread(curve)
            if current > 1e-6:
                stretch = float(widths[row]) / current

        def place(shift: float) -> np.ndarray:
            # Stretch about the curve's own median, then slide it, so the two
            # adjustments do not fight each other.
            source = (LOG_SUPPORTS - centre) * stretch + centre + shift
            return np.interp(LOG_SUPPORTS, source, curve, left=0.0, right=100.0)

        # A target near either end of the support range pushes part of the
        # curve outside it, where interpolation clamps to 0 or 100.  That
        # truncation drags the median back toward the centre, so solve for the
        # shift whose *result* has the requested d50 rather than assuming it.
        shift = want - centre
        for _ in range(12):
            placed = place(shift)
            achieved = _log_d50(placed[None, :])[0]
            error = want - achieved
            if abs(error) < 1e-4:
                break
            shift += error
        out[row] = place(shift)
    return project_valid(out)


def curves_from_template(
    readings: dict,
    sample_ids,
    training_curves: np.ndarray,
) -> np.ndarray:
    """Build curves by warping a real training curve onto each reading.

    The lognormal used by :func:`curves_from_readings` is a poor model of a
    soil: real gradations saturate at a true maximum particle size while a
    lognormal only approaches 100 asymptotically.  Given nothing but a d50 and
    a spread - exactly what a reading supplies - reproducing the 24 training
    curves costs 14.30 EMD with a lognormal and clears 10 for 7 of them.
    Taking the training soil whose own d50 is nearest the reading and warping
    it onto that d50 and spread costs 8.35 and clears 10 for 18.

    Selecting the template by d50 rather than by texture matters: an earlier
    version took its shape from the neighbour model and scored worse than the
    lognormal on the leaderboard.  Nearness in grain size is what makes two
    gradations resemble each other.
    """
    from .models import _log_d50

    template_d50 = _log_d50(np.asarray(training_curves, dtype=float))
    out = []
    for sample_id in pd.Series(sample_ids).astype(str):
        entry = readings.get(sample_id)
        if entry is None:
            raise KeyError(f"no visual reading recorded for {sample_id!r}")
        want = np.log10(float(entry["d50_mm"]))
        spread = 2.0 * float(entry["sigma"])
        nearest = int(np.argmin(np.abs(template_d50 - want)))
        out.append(
            warp_to_reading(
                training_curves[nearest][None, :], [10.0**want], [spread]
            )[0]
        )
    return project_valid(np.array(out))
