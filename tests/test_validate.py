from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soilgsd.constants import ID_COLUMN, SUBMISSION_COLUMNS, TARGET_COLUMNS
from soilgsd.validate import validate_submission, validate_submission_file


def _good(n: int = 4) -> pd.DataFrame:
    curve = np.linspace(0.0, 100.0, 11)
    frame = pd.DataFrame(np.tile(curve, (n, 1)), columns=list(TARGET_COLUMNS))
    frame.insert(0, ID_COLUMN, [f"s{i}" for i in range(n)])
    return frame.loc[:, list(SUBMISSION_COLUMNS)]


def test_a_well_formed_submission_passes():
    report = validate_submission(_good())
    assert report.ok, report.render()
    assert "PASS" in report.render()


def test_out_of_range_values_are_caught():
    frame = _good()
    frame.loc[0, "0.002"] = -1.0
    assert any("must lie in" in issue for issue in validate_submission(frame).issues)


def test_non_monotone_rows_are_caught():
    frame = _good()
    frame.loc[1, "0.63"] = 0.0
    issues = validate_submission(frame).issues
    assert any("non-decreasing" in issue for issue in issues)


def test_terminal_value_must_be_one_hundred():
    frame = _good()
    frame.loc[2, "200"] = 99.9
    assert any("end at 100" in issue for issue in validate_submission(frame).issues)


def test_duplicate_ids_are_caught():
    frame = _good()
    frame.loc[1, ID_COLUMN] = frame.loc[0, ID_COLUMN]
    assert any("duplicate" in issue for issue in validate_submission(frame).issues)


def test_column_order_must_match_exactly():
    frame = _good()[[ID_COLUMN, *reversed(TARGET_COLUMNS)]]
    assert any("columns must be exactly" in issue for issue in validate_submission(frame).issues)


def test_template_mismatch_is_reported():
    template = _good(5)
    report = validate_submission(_good(4), template)
    assert any("row count" in issue for issue in report.issues)
    assert any("missing from submission" in issue for issue in report.issues)


def test_file_roundtrip(tmp_path):
    path = tmp_path / "submission.csv"
    _good().to_csv(path, index=False)
    assert validate_submission_file(path).ok
