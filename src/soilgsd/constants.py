"""Fixed facts about the competition target space.

The target is a cumulative grain-size distribution (percent passing) sampled at
eleven fixed sieve diameters, from clay through the cobble boundary.
"""

from __future__ import annotations

import numpy as np

ID_COLUMN = "sample_id"

#: The eleven support diameters in millimetres.
SUPPORTS_MM: tuple[float, ...] = (
    0.002,
    0.0063,
    0.02,
    0.063,
    0.2,
    0.63,
    2.0,
    6.3,
    20.0,
    63.0,
    200.0,
)

#: Submission column names, exactly as they appear in ``sample_submission.csv``.
TARGET_COLUMNS: tuple[str, ...] = (
    "0.002",
    "0.0063",
    "0.02",
    "0.063",
    "0.2",
    "0.63",
    "2",
    "6.3",
    "20",
    "63",
    "200",
)

SUBMISSION_COLUMNS: tuple[str, ...] = (ID_COLUMN, *TARGET_COLUMNS)

N_SUPPORTS = len(SUPPORTS_MM)

#: log10 of each support, i.e. the axis the metric integrates over.
LOG_SUPPORTS = np.log10(np.asarray(SUPPORTS_MM, dtype=np.float64))

#: Width of each of the ten adjacent support intervals on the log10 axis.
#: These are the metric's weights; they sum to log10(200 / 0.002) = 5.
LOG_WIDTHS = np.diff(LOG_SUPPORTS)

#: Worst achievable score: 100 percentage points of error over the whole axis.
MAX_SCORE = 100.0 * float(LOG_WIDTHS.sum())

#: Percent passing is bounded, and the coarsest support is 100 by construction.
VALUE_MIN = 0.0
VALUE_MAX = 100.0
