"""End-to-end steps: index photos, cache features, cross-validate, submit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, SUBMISSION_COLUMNS, TARGET_COLUMNS
from .curves import project_valid
from .data import DataPaths, build_photo_index, load_labels, load_ppm, load_sample_submission
from .evaluate import CVResult, cross_validate
from .features import (
    FeatureConfig,
    aggregate_to_samples,
    extract_photo_features,
    select_features,
)
from .models import build_model
from .validate import validate_submission

__all__ = ["Settings", "load_settings", "prepare_features", "run_cv", "make_submission"]


@dataclass
class Settings:
    data_root: str = "data/raw"
    artifacts: str = "artifacts"
    model: str = "knn"
    feature_set: str = "texture"
    n_splits: int = 24
    group_pattern: str | None = None
    features: FeatureConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.features is None:
            self.features = FeatureConfig()
        elif isinstance(self.features, dict):
            self.features = FeatureConfig(**self.features)

    @property
    def artifact_dir(self) -> Path:
        path = Path(self.artifacts)
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_settings(path: str | Path | None = None, **overrides) -> Settings:
    """Load settings from a YAML file, with keyword overrides applied on top."""
    payload: dict = {}
    if path is not None:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return Settings(**payload)


def _feature_cache_path(settings: Settings, split: str) -> Path:
    config = settings.features
    tag = (
        f"ppm{config.target_ppm:g}_crop{config.crop_px}x{config.crop_grid}"
        f"_lv{config.n_levels}_soil{int(config.crop_to_soil)}"
    )
    return settings.artifact_dir / "features" / f"{split}_{tag}.csv"


def prepare_features(
    settings: Settings,
    *,
    split: str = "train",
    refresh: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Build (or reuse) the cached sample-level feature table for one split.

    Photo features are extracted once per photo and then pooled per sample.  The
    cache key encodes the feature settings, so changing ``target_ppm`` or the
    crop layout produces a new cache rather than silently reusing a stale one.
    """
    cache = _feature_cache_path(settings, split)
    if cache.exists() and not refresh:
        frame = pd.read_csv(cache)
        frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
        return frame

    paths = DataPaths.discover(settings.data_root)
    if split == "train":
        sample_ids = load_labels(paths.labels)[ID_COLUMN]
        photo_dir = paths.train_photos
    elif split == "test":
        sample_ids = load_sample_submission(paths.sample_submission)[ID_COLUMN]
        photo_dir = paths.test_photos
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    ppm = load_ppm(paths.ppm) if paths.ppm else None
    index = build_photo_index(
        photo_dir, sample_ids, ppm=ppm, default_ppm=settings.features.default_ppm
    )
    unmatched = index.attrs.get("unmatched", [])
    if unmatched:
        print(
            f"[warn] {len(unmatched)} {split} photo(s) could not be matched to a sample_id, "
            f"e.g. {unmatched[:3]}"
        )
    if index.empty:
        raise RuntimeError(f"no {split} photo could be matched to a sample_id in {photo_dir}")

    per_photo = extract_photo_features(index, settings.features, progress=progress)
    failures = per_photo.attrs.get("failures", [])
    if failures:
        print(f"[warn] {len(failures)} photo(s) failed to load, e.g. {failures[:2]}")

    frame = aggregate_to_samples(per_photo)
    missing = sorted(set(sample_ids.astype(str)) - set(frame[ID_COLUMN]))
    if missing:
        print(f"[warn] {len(missing)} {split} sample(s) have no usable photo, e.g. {missing[:3]}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def run_cv(settings: Settings, *, model: str | None = None, refresh: bool = False) -> CVResult:
    """Cross-validate one model and write its out-of-fold predictions to artifacts."""
    name = model or settings.model
    paths = DataPaths.discover(settings.data_root)
    labels = load_labels(paths.labels)
    features = prepare_features(settings, split="train", refresh=refresh)

    result = cross_validate(
        lambda: build_model(name),
        features[[ID_COLUMN, *select_features(features.columns, settings.feature_set)]],
        labels,
        n_splits=settings.n_splits,
        group_pattern=settings.group_pattern,
        model_name=name,
    )

    reports = settings.artifact_dir / "cv"
    reports.mkdir(parents=True, exist_ok=True)
    result.oof_predictions.to_csv(reports / f"oof_{name.replace(':', '_')}.csv", index=False)
    result.per_sample.to_csv(reports / f"per_sample_{name.replace(':', '_')}.csv", index=False)
    summary = {
        "model": name,
        "score": result.score,
        "fold_scores": result.fold_scores,
        "n_samples": int(len(result.per_sample)),
        "feature_set": settings.feature_set,
        "n_features": len(select_features(features.columns, settings.feature_set)),
        "per_support_mae": {k: float(v) for k, v in result.per_support_mae.items()},
        "features": asdict(settings.features),
    }
    with open(reports / f"summary_{name.replace(':', '_')}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return result


def make_submission(
    settings: Settings,
    *,
    model: str | None = None,
    out_path: str | Path | None = None,
    refresh: bool = False,
) -> Path:
    """Fit on all training samples, predict the test set, and write a valid CSV.

    Test samples with no usable photo fall back to the training median curve,
    so the submission always covers every row of the template.
    """
    name = model or settings.model
    paths = DataPaths.discover(settings.data_root)
    labels = load_labels(paths.labels)
    template = load_sample_submission(paths.sample_submission)

    train_features = prepare_features(settings, split="train", refresh=refresh)
    test_features = prepare_features(settings, split="test", refresh=refresh)

    merged = labels.merge(train_features, on=ID_COLUMN, how="inner", validate="one_to_one")
    feature_columns = select_features(train_features.columns, settings.feature_set)
    fitted = build_model(name).fit(merged[feature_columns], merged[list(TARGET_COLUMNS)])

    # Reindex onto the template so every required id is present and in order;
    # rows with no features become NaN and are filled with the fallback below.
    aligned = test_features.set_index(ID_COLUMN).reindex(template[ID_COLUMN]).reset_index()
    usable = aligned[feature_columns].notna().all(axis=1).to_numpy()

    predictions = np.empty((len(template), len(TARGET_COLUMNS)), dtype=np.float64)
    fallback = build_model("constant").fit(None, merged[list(TARGET_COLUMNS)]).curve_
    predictions[~usable] = fallback
    if usable.any():
        predictions[usable] = fitted.predict(aligned.loc[usable, feature_columns])
    if (~usable).any():
        print(
            f"[warn] {int((~usable).sum())} test sample(s) had no usable photo; "
            "filled with the training median curve"
        )

    submission = pd.DataFrame(project_valid(predictions), columns=list(TARGET_COLUMNS)).round(4)
    submission.insert(0, ID_COLUMN, template[ID_COLUMN].to_numpy())
    submission = submission.loc[:, list(SUBMISSION_COLUMNS)]

    report = validate_submission(submission, template)
    if not report.ok:
        raise RuntimeError(f"refusing to write an invalid submission:\n{report.render()}")

    destination = Path(out_path) if out_path else (
        settings.artifact_dir / "submissions" / f"{name.replace(':', '_')}.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    return destination
