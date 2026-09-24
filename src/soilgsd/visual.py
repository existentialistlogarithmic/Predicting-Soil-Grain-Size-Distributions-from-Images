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

__all__ = ["load_readings", "curves_from_readings", "blend_with_model"]

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
