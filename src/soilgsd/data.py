"""Loading the official competition files and indexing the photo folders.

The archive published by the host contains::

    Training_labels_updated.csv      one row per training sample, eleven targets
    sample_submission.csv            the test sample ids and the column order
    ppm_updated.csv                  camera scale (pixels per millimetre)
    Training-All_Photos_updated/     training photographs
    Test_All_Photos/                 test photographs

A sample is photographed more than once, so photo filenames have to be mapped
back onto sample ids.  Rather than hard-code one naming scheme, the index below
matches each filename against the known sample ids and reports what it could
not resolve, which keeps the pipeline honest if the host renames things.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, SUBMISSION_COLUMNS, TARGET_COLUMNS

__all__ = [
    "DataPaths",
    "load_labels",
    "load_sample_submission",
    "load_ppm",
    "build_photo_index",
    "IMAGE_SUFFIXES",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

_LABELS_CANDIDATES = ("Training_labels_updated.csv", "Training_labels.csv", "train.csv")
_PPM_CANDIDATES = ("ppm_updated.csv", "ppm.csv")
_TRAIN_PHOTO_CANDIDATES = ("Training-All_Photos_updated", "Training_All_Photos", "train_photos")
_TEST_PHOTO_CANDIDATES = ("Test_All_Photos", "test_photos")


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True)
class DataPaths:
    """Resolved locations of the competition files under a single root."""

    root: Path
    labels: Path
    sample_submission: Path
    train_photos: Path
    test_photos: Path
    ppm: Path | None = None

    @classmethod
    def discover(cls, root: str | Path) -> "DataPaths":
        """Locate the official files under ``root``, raising if any are missing."""
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"data root {root} does not exist; run `make download` or see README"
            )
        labels = _first_existing(root, _LABELS_CANDIDATES)
        submission = _first_existing(root, ("sample_submission.csv",))
        train_photos = _first_existing(root, _TRAIN_PHOTO_CANDIDATES)
        test_photos = _first_existing(root, _TEST_PHOTO_CANDIDATES)
        missing = [
            name
            for name, value in (
                ("training labels csv", labels),
                ("sample_submission.csv", submission),
                ("training photo folder", train_photos),
                ("test photo folder", test_photos),
            )
            if value is None
        ]
        if missing:
            listing = ", ".join(sorted(p.name for p in root.iterdir())) or "<empty>"
            raise FileNotFoundError(
                f"missing {missing} under {root}. Found: {listing}"
            )
        assert labels and submission and train_photos and test_photos
        return cls(
            root=root,
            labels=labels,
            sample_submission=submission,
            train_photos=train_photos,
            test_photos=test_photos,
            ppm=_first_existing(root, _PPM_CANDIDATES),
        )


def _coerce_targets(frame: pd.DataFrame, *, source: Path) -> pd.DataFrame:
    missing = [column for column in TARGET_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source} is missing target columns {missing}; found {list(frame.columns)}"
        )
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"{source} has no {ID_COLUMN} column; found {list(frame.columns)}")
    out = frame.loc[:, list(SUBMISSION_COLUMNS)].copy()
    out[ID_COLUMN] = out[ID_COLUMN].astype(str)
    out[list(TARGET_COLUMNS)] = out[list(TARGET_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    ).astype(np.float64)
    return out


def load_labels(path: str | Path) -> pd.DataFrame:
    """Load the training labels as ``sample_id`` plus the eleven target columns."""
    path = Path(path)
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    return _coerce_targets(frame, source=path).reset_index(drop=True)


def load_sample_submission(path: str | Path) -> pd.DataFrame:
    """Load the submission template; its row order is the order Kaggle expects."""
    path = Path(path)
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    return _coerce_targets(frame, source=path).reset_index(drop=True)


def load_ppm(path: str | Path) -> pd.DataFrame:
    """Load the camera-scale table as ``key`` / ``ppm``.

    The host ships pixels-per-millimetre keyed by photo or by camera.  The
    column names are not fixed, so the first column that looks like a scale is
    taken as ``ppm`` and the first string-like column as the join ``key``.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]

    ppm_column = next(
        (
            column
            for column in frame.columns
            if re.search(r"ppm|pixels?[_ ]?per[_ ]?mm|scale|resolution", column, re.I)
        ),
        None,
    )
    if ppm_column is None:
        numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        if not numeric:
            raise ValueError(f"{path} has no numeric column to use as ppm")
        ppm_column = numeric[-1]

    key_column = next(
        (
            column
            for column in frame.columns
            if column != ppm_column
            and re.search(r"photo|image|file|name|camera|id", column, re.I)
        ),
        None,
    )
    if key_column is None:
        key_column = next((c for c in frame.columns if c != ppm_column), None)
    if key_column is None:
        raise ValueError(f"{path} has no key column to join on")

    out = pd.DataFrame(
        {
            "key": frame[key_column].astype(str).str.strip(),
            "ppm": pd.to_numeric(frame[ppm_column], errors="coerce"),
        }
    )
    out["key_stem"] = out["key"].map(lambda value: Path(value).stem.lower())
    return out.dropna(subset=["ppm"]).reset_index(drop=True)


def _match_sample_id(stem: str, sample_ids: list[str]) -> str | None:
    """Find which sample id a photo filename belongs to.

    Tries an exact stem match, then the longest sample id that appears in the
    stem.  Preferring the longest match stops ``S1`` from stealing photos that
    belong to ``S12``.
    """
    lowered = stem.lower()
    best: str | None = None
    for sample_id in sample_ids:
        candidate = sample_id.lower()
        if lowered == candidate:
            return sample_id
        if candidate in lowered and (best is None or len(sample_id) > len(best)):
            best = sample_id
    return best


def build_photo_index(
    photo_dir: str | Path,
    sample_ids: list[str] | pd.Series,
    *,
    ppm: pd.DataFrame | None = None,
    default_ppm: float | None = None,
) -> pd.DataFrame:
    """Index every photograph under ``photo_dir`` and attach its sample id and scale.

    Returns one row per photo with ``sample_id``, ``photo_id``, ``path`` and
    ``ppm``.  Photos whose sample id cannot be resolved are dropped, and the
    count is reported through the ``unmatched`` attribute on the result's
    ``attrs`` so callers can surface it rather than silently lose data.
    """
    photo_dir = Path(photo_dir)
    if not photo_dir.is_dir():
        raise FileNotFoundError(f"photo folder {photo_dir} does not exist")

    known = [str(value) for value in pd.Series(sample_ids).astype(str).unique()]
    # Longest first so the substring search below settles ties deterministically.
    known.sort(key=len, reverse=True)

    records: list[dict[str, object]] = []
    unmatched: list[str] = []
    for path in sorted(photo_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        sample_id = _match_sample_id(path.stem, known)
        if sample_id is None:
            # A photo may instead sit in a folder named after its sample.
            sample_id = _match_sample_id(path.parent.name, known)
        if sample_id is None:
            unmatched.append(str(path.relative_to(photo_dir)))
            continue
        records.append(
            {
                ID_COLUMN: sample_id,
                "photo_id": path.stem,
                "path": str(path),
                "ppm": np.nan,
            }
        )

    index = pd.DataFrame.from_records(
        records, columns=[ID_COLUMN, "photo_id", "path", "ppm"]
    )

    if ppm is not None and not index.empty:
        lookup = dict(zip(ppm["key_stem"], ppm["ppm"]))
        by_key = dict(zip(ppm["key"].str.lower(), ppm["ppm"]))
        resolved = []
        for photo_id, sample_id in zip(index["photo_id"], index[ID_COLUMN]):
            value = lookup.get(photo_id.lower())
            if value is None:
                value = by_key.get(photo_id.lower())
            if value is None:
                # ppm may be keyed by sample or by camera rather than by photo.
                value = lookup.get(str(sample_id).lower())
            resolved.append(np.nan if value is None else float(value))
        index["ppm"] = resolved

    if default_ppm is not None:
        index["ppm"] = index["ppm"].fillna(float(default_ppm))

    index.attrs["unmatched"] = unmatched
    index.attrs["photo_dir"] = str(photo_dir)
    return index.reset_index(drop=True)
