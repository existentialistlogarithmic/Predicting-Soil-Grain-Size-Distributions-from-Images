"""Tools for the Kaggle competition *Predicting Soil Grain Size Distributions from Images*.

The competition asks for a cumulative grain size distribution at eleven fixed
sieve diameters, scored by a log-diameter-weighted Earth Mover's Distance.  See
:mod:`soilgsd.metric` for what that metric implies for modelling, and
``docs/COMPETITION.md`` for the competition's fixed facts.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "constants",
    "curves",
    "data",
    "evaluate",
    "features",
    "metric",
    "models",
    "pipeline",
    "synthetic",
    "validate",
]
