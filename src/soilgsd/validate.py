"""Check a submission the way Kaggle will, before spending one of five daily slots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, SUBMISSION_COLUMNS, TARGET_COLUMNS, VALUE_MAX, VALUE_MIN

__all__ = ["SubmissionReport", "validate_submission", "validate_submission_file"]


@dataclass
class SubmissionReport:
    path: str | None
    n_rows: int
    n_expected: int | None
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def render(self) -> str:
        header = f"{self.path or '<frame>'}: {self.n_rows} rows"
        if self.n_expected is not None:
            header += f" (template has {self.n_expected})"
        if self.ok:
            return f"{header}\nPASS - submission is structurally valid."
        lines = "\n".join(f"  - {issue}" for issue in self.issues)
        return f"{header}\nFAIL - {len(self.issues)} issue(s):\n{lines}"


def validate_submission(
    submission: pd.DataFrame,
    template: pd.DataFrame | None = None,
    *,
    path: str | None = None,
    tolerance: float = 1e-8,
) -> SubmissionReport:
    """Return every structural problem Kaggle would reject the file for."""
    issues: list[str] = []

    if list(submission.columns) != list(SUBMISSION_COLUMNS):
        issues.append(
            f"columns must be exactly {list(SUBMISSION_COLUMNS)}, got {list(submission.columns)}"
        )
    if ID_COLUMN not in submission.columns:
        return SubmissionReport(path, len(submission), None, issues + ["no sample_id column"])

    ids = submission[ID_COLUMN].astype(str)
    if ids.duplicated().any():
        duplicates = sorted(ids[ids.duplicated()].unique())[:5]
        issues.append(f"duplicate sample_id values, e.g. {duplicates}")

    present = [column for column in TARGET_COLUMNS if column in submission.columns]
    numeric = submission.reindex(columns=list(TARGET_COLUMNS)).apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        issues.append(f"non-numeric or missing values in {bad}")

    values = numeric.to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    if values.size and finite.all():
        if values.min() < VALUE_MIN - tolerance or values.max() > VALUE_MAX + tolerance:
            issues.append(
                f"values must lie in [{VALUE_MIN:g}, {VALUE_MAX:g}]; "
                f"observed [{values.min():.4g}, {values.max():.4g}]"
            )
        violations = np.diff(values, axis=1) < -tolerance
        if violations.any():
            rows = np.flatnonzero(violations.any(axis=1))
            issues.append(
                f"{rows.size} row(s) are not non-decreasing, e.g. sample_id "
                f"{ids.iloc[rows[:3]].tolist()}"
            )
        terminal_off = np.abs(values[:, -1] - VALUE_MAX) > 1e-6
        if terminal_off.any():
            rows = np.flatnonzero(terminal_off)
            issues.append(
                f"{rows.size} row(s) do not end at 100 in the '200' column, e.g. sample_id "
                f"{ids.iloc[rows[:3]].tolist()}"
            )
    if len(present) != len(TARGET_COLUMNS):
        issues.append(
            f"missing target columns {[c for c in TARGET_COLUMNS if c not in present]}"
        )

    n_expected = None
    if template is not None:
        n_expected = len(template)
        if len(submission) != n_expected:
            issues.append(f"row count {len(submission)} differs from template {n_expected}")
        template_ids = set(template[ID_COLUMN].astype(str))
        submitted_ids = set(ids)
        if template_ids != submitted_ids:
            missing = sorted(template_ids - submitted_ids)[:5]
            extra = sorted(submitted_ids - template_ids)[:5]
            if missing:
                issues.append(f"sample_id values missing from submission, e.g. {missing}")
            if extra:
                issues.append(f"sample_id values not in the template, e.g. {extra}")

    return SubmissionReport(path, len(submission), n_expected, issues)


def validate_submission_file(
    path: str | Path, template_path: str | Path | None = None
) -> SubmissionReport:
    """Validate a CSV on disk, optionally against ``sample_submission.csv``."""
    submission = pd.read_csv(path)
    template = pd.read_csv(template_path) if template_path else None
    return validate_submission(submission, template, path=str(path))
