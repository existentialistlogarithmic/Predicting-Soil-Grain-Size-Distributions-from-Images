"""Grouped cross-validation against the competition metric.

Photos of one soil must never straddle a fold: features are aggregated per
sample before modelling, but several samples can come from the same physical
soil, and a random split would then leak a near-duplicate into validation and
flatter the model.  ``group_of`` derives that soil key, defaulting to the
sample id itself when nothing more specific is configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, TARGET_COLUMNS
from .metric import weighted_emd, weighted_emd_per_sample
from .models import CurveModel

__all__ = ["CVResult", "derive_groups", "cross_validate"]


@dataclass
class CVResult:
    model_name: str
    score: float
    fold_scores: list[float]
    per_sample: pd.DataFrame
    oof_predictions: pd.DataFrame
    per_support_mae: pd.Series = field(default_factory=pd.Series)

    def render(self) -> str:
        folds = ", ".join(f"{value:.3f}" for value in self.fold_scores)
        spread = np.std(self.fold_scores) if len(self.fold_scores) > 1 else 0.0
        lines = [
            f"{self.model_name}: OOF weighted EMD = {self.score:.4f} "
            f"(fold sd {spread:.3f})",
            f"  folds: {folds}",
        ]
        if not self.per_support_mae.empty:
            worst = self.per_support_mae.sort_values(ascending=False).head(3)
            detail = ", ".join(f"{name} mm: {value:.2f}" for name, value in worst.items())
            lines.append(f"  worst supports (MAE in % passing): {detail}")
        return "\n".join(lines)


def derive_groups(sample_ids: pd.Series, pattern: str | None = None) -> pd.Series:
    """Map sample ids onto soil groups for grouped CV.

    With no ``pattern`` each sample is its own group.  Given a regex with one
    capturing group, the captured text becomes the group key, so a scheme like
    ``SOIL07_rep2`` can be collapsed with ``r'^([^_]+)'``.
    """
    ids = sample_ids.astype(str)
    if not pattern:
        return ids
    compiled = re.compile(pattern)

    def extract(value: str) -> str:
        match = compiled.search(value)
        if match is None:
            return value
        return match.group(1) if match.groups() else match.group(0)

    return ids.map(extract)


def cross_validate(
    model_factory,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_splits: int = 5,
    group_pattern: str | None = None,
    model_name: str | None = None,
) -> CVResult:
    """Run grouped K-fold CV and return out-of-fold predictions and scores.

    ``model_factory`` is called once per fold so no state survives between
    folds; pass a class or a zero-argument lambda.
    """
    from sklearn.model_selection import GroupKFold

    merged = labels.merge(features, on=ID_COLUMN, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("no sample_id is present in both the labels and the features")

    feature_columns = [
        column
        for column in features.columns
        if column != ID_COLUMN and pd.api.types.is_numeric_dtype(features[column])
    ]
    matrix = merged[feature_columns]
    curves = merged[list(TARGET_COLUMNS)]
    groups = derive_groups(merged[ID_COLUMN], group_pattern)

    n_groups = groups.nunique()
    splits = int(min(n_splits, n_groups))
    if splits < 2:
        raise ValueError(f"need at least 2 groups for CV, found {n_groups}")

    oof = np.full((len(merged), len(TARGET_COLUMNS)), np.nan)
    fold_scores: list[float] = []
    for train_index, test_index in GroupKFold(n_splits=splits).split(matrix, curves, groups):
        model: CurveModel = model_factory() if callable(model_factory) else model_factory
        model.fit(matrix.iloc[train_index], curves.iloc[train_index])
        predicted = model.predict(matrix.iloc[test_index])
        oof[test_index] = predicted
        fold_scores.append(weighted_emd(curves.iloc[test_index], predicted))

    per_sample_score = weighted_emd_per_sample(curves, oof)
    per_sample = pd.DataFrame(
        {
            ID_COLUMN: merged[ID_COLUMN].to_numpy(),
            "group": groups.to_numpy(),
            "emd": per_sample_score,
        }
    ).sort_values("emd", ascending=False, ignore_index=True)

    oof_frame = pd.DataFrame(oof, columns=list(TARGET_COLUMNS))
    oof_frame.insert(0, ID_COLUMN, merged[ID_COLUMN].to_numpy())

    absolute_error = np.abs(curves.to_numpy(dtype=np.float64) - oof)
    per_support = pd.Series(absolute_error.mean(axis=0), index=list(TARGET_COLUMNS))

    return CVResult(
        model_name=model_name or getattr(model_factory, "name", "model"),
        score=float(np.mean(per_sample_score)),
        fold_scores=fold_scores,
        per_sample=per_sample,
        oof_predictions=oof_frame,
        per_support_mae=per_support,
    )
