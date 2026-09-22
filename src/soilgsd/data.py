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
    "normalise_key",
    "resolve_camera",
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


def normalise_key(value: str) -> str:
    """Fold a filename or identifier down to comparable alphanumerics.

    The archive is inconsistent in ways that matter for joining: ``iPhone16``
    and ``iPhone_16`` are one camera, and one test sample reaches us as
    ``HPC_M#U00fcnster_BS6_9,0-10m`` while ``sample_submission.csv`` spells it
    ``HPC_Muenster_BS6_9_0-10m``.  Decoding the ``#U00xx`` escape, expanding
    German umlauts the way the host does, and dropping punctuation makes both
    spellings land on the same key.
    """
    text = re.sub(
        r"#U([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        str(value),
    )
    for source, target in (
        ("ä", "ae"), ("ö", "oe"), ("ü", "ue"),
        ("Ä", "ae"), ("Ö", "oe"), ("Ü", "ue"),
        ("ß", "ss"),
    ):
        text = text.replace(source, target)
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def load_ppm(path: str | Path) -> pd.DataFrame:
    """Load the camera scale table.

    ``ppm_updated.csv`` gives pixels per millimetre *for a stated reference
    resolution*, keyed by phone and camera model rather than by photograph.
    Both facts matter: the scale has to be matched to a photo via its camera
    name, and then corrected for the resolution the file was actually
    delivered at (see :func:`build_photo_index`).
    """
    path = Path(path)
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]

    def pick(pattern: str, exclude: set[str] | None = None) -> str | None:
        for column in frame.columns:
            if column in (exclude or set()):
                continue
            if re.search(pattern, column, re.I):
                return column
        return None

    ppm_column = pick(r"^ppm$|pixels?[_ ]?per[_ ]?mm|scale")
    if ppm_column is None:
        numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        if not numeric:
            raise ValueError(f"{path} has no numeric column to use as ppm")
        ppm_column = numeric[-1]
    width_column = pick(r"width")
    height_column = pick(r"height")

    records: list[dict[str, object]] = []
    name_columns = [
        column
        for column in frame.columns
        if column not in {ppm_column, width_column, height_column}
        and not pd.api.types.is_numeric_dtype(frame[column])
    ]
    for _, row in frame.iterrows():
        scale = pd.to_numeric(row[ppm_column], errors="coerce")
        if not np.isfinite(scale):
            continue
        reference = np.nan
        if width_column and height_column:
            width = pd.to_numeric(row[width_column], errors="coerce")
            height = pd.to_numeric(row[height_column], errors="coerce")
            if np.isfinite(width) and np.isfinite(height):
                reference = float(max(width, height))
        # One camera can be spelled several ways across the columns; index every
        # spelling so a filename can match on whichever one it happens to use.
        for column in name_columns:
            key = normalise_key(row[column])
            if key:
                records.append(
                    {
                        "camera_key": key,
                        "label": str(row[column]).strip(),
                        "ppm": float(scale),
                        "reference_long_side": reference,
                    }
                )

    if not records:
        raise ValueError(f"{path} yielded no usable camera rows")
    out = pd.DataFrame.from_records(records).drop_duplicates(subset="camera_key")
    return out.reset_index(drop=True)


def resolve_camera(stem: str, ppm_table: pd.DataFrame) -> pd.Series | None:
    """Find which camera took a photo, from its filename.

    Filenames lead with the camera (``Samsung_A52_H030_01``), so the match is
    on prefix.  The longest key wins, or ``Motorola_Edge`` would swallow the
    photos belonging to ``Motorola_Edge_60_fusion``.
    """
    key = normalise_key(stem)
    best: pd.Series | None = None
    for _, row in ppm_table.iterrows():
        candidate = str(row["camera_key"])
        if not candidate:
            continue
        hit = key.startswith(candidate) or candidate in key
        if hit and (best is None or len(candidate) > len(str(best["camera_key"]))):
            best = row
    return best


def _match_sample_id(stem: str, normalised_ids: list[tuple[str, str]]) -> str | None:
    """Find which sample a photo filename belongs to.

    Both sides are folded through :func:`normalise_key` first, which is what
    lets ``iPhone14_HPC_M#U00fcnster_BS6_9,0-10m`` find the sample spelled
    ``HPC_Muenster_BS6_9_0-10m``.  The longest matching id wins so a short code
    cannot steal photos from a longer one that contains it.
    """
    key = normalise_key(stem)
    best: str | None = None
    best_length = 0
    for sample_id, candidate in normalised_ids:
        if candidate and candidate in key and len(candidate) > best_length:
            best, best_length = sample_id, len(candidate)
    return best


def _image_long_side(path: Path) -> float:
    """Longest side of an image in pixels, read from the header alone."""
    from PIL import Image

    with Image.open(path) as handle:
        return float(max(handle.size))


def build_photo_index(
    photo_dir: str | Path,
    sample_ids: list[str] | pd.Series,
    *,
    ppm: pd.DataFrame | None = None,
    default_ppm: float | None = None,
) -> pd.DataFrame:
    """Index every photograph under ``photo_dir`` with its sample id and true scale.

    The scale needs care.  ``ppm_updated.csv`` quotes pixels per millimetre at
    a reference resolution, but most training photos were delivered downscaled:
    a Samsung A52 frame is shipped at 1599 px across, not the 9248 px the table
    assumes, so its real scale is 4.55 ppm rather than 26.33.  Test photos are
    all at native resolution.  Taking the quoted number at face value would
    therefore read training texture up to 2.5 octaves finer than it really is
    while reading the test set correctly — a silent train/test mismatch.  The
    correction is simply the ratio of delivered to reference resolution.

    Returns one row per photo with ``sample_id``, ``photo_id``, ``path``,
    ``ppm`` (corrected), ``camera`` and ``resolution_scale``.  Photos whose
    sample cannot be resolved are dropped and listed in ``attrs['unmatched']``.
    """
    photo_dir = Path(photo_dir)
    if not photo_dir.is_dir():
        raise FileNotFoundError(f"photo folder {photo_dir} does not exist")

    normalised_ids = [
        (str(value), normalise_key(str(value)))
        for value in pd.Series(sample_ids).astype(str).unique()
    ]

    records: list[dict[str, object]] = []
    unmatched: list[str] = []
    no_camera: list[str] = []
    for path in sorted(photo_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        sample_id = _match_sample_id(path.stem, normalised_ids)
        if sample_id is None:
            sample_id = _match_sample_id(path.parent.name, normalised_ids)
        if sample_id is None:
            unmatched.append(str(path.relative_to(photo_dir)))
            continue

        scale = float("nan")
        camera = ""
        resolution_scale = float("nan")
        if ppm is not None:
            row = resolve_camera(path.stem, ppm)
            if row is None:
                no_camera.append(path.name)
            else:
                camera = str(row["label"])
                scale = float(row["ppm"])
                reference = row["reference_long_side"]
                if reference and np.isfinite(reference) and reference > 0:
                    resolution_scale = _image_long_side(path) / float(reference)
                    scale *= resolution_scale

        records.append(
            {
                ID_COLUMN: sample_id,
                "photo_id": path.stem,
                "path": str(path),
                "ppm": scale,
                "camera": camera,
                "resolution_scale": resolution_scale,
            }
        )

    index = pd.DataFrame.from_records(
        records,
        columns=[ID_COLUMN, "photo_id", "path", "ppm", "camera", "resolution_scale"],
    )
    if default_ppm is not None and not index.empty:
        index["ppm"] = index["ppm"].fillna(float(default_ppm))

    index.attrs["unmatched"] = unmatched
    index.attrs["no_camera"] = no_camera
    index.attrs["photo_dir"] = str(photo_dir)
    return index.reset_index(drop=True)
