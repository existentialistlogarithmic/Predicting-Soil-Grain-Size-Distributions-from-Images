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

    This is the piece that was missing.  The neighbour model is good at *shape*
    - it picks the training soils whose texture matches, and their curves carry
    a realistic gradation - but it cannot place that shape correctly, because
    pooling training curves cannot reach past the training range.  A diameter
    read off a scale bar places it, and sliding along log-diameter has no
    ceiling.

    Measured by leave-one-out with the true d50 standing in for a reading, the
    neighbour model goes from 37.08 to 17.19.  With a reading accurate to 0.1
    decades, about +/-26%, it is 18.91; at 0.3 decades, a factor of two, 26.33;
    the two break even near 0.5 decades.  So the shift is worth making as long
    as the diameter is known to better than a factor of about three, which
    reading it off a bar comfortably is.
    """
    from .models import _log_d50

    curves = np.atleast_2d(np.asarray(curves, dtype=float))
    targets = np.log10(np.atleast_1d(np.asarray(target_d50_mm, dtype=float)))
    if len(targets) != len(curves):
        raise ValueError(f"got {len(curves)} curves and {len(targets)} targets")

    current = _log_d50(curves)
    out = np.empty_like(curves)
    for row, (curve, have, want) in enumerate(zip(curves, current, targets)):
        out[row] = np.interp(
            LOG_SUPPORTS, LOG_SUPPORTS + (want - have), curve, left=0.0, right=100.0
        )
    return project_valid(out)


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
        # Stretch about the curve's own median, then slide that median onto the
        # reading, so the two adjustments do not fight each other.
        source = (LOG_SUPPORTS - centre) * stretch + centre + (want - centre)
        out[row] = np.interp(LOG_SUPPORTS, source, curve, left=0.0, right=100.0)
    return project_valid(out)
