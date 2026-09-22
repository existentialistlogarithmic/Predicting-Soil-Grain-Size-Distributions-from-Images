from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from soilgsd.cli import main
from soilgsd.constants import ID_COLUMN, SUBMISSION_COLUMNS
from soilgsd.features import FeatureConfig
from soilgsd.pipeline import Settings, load_settings, make_submission, prepare_features, run_cv
from soilgsd.validate import validate_submission_file


@pytest.fixture
def settings(synthetic_root, tmp_path) -> Settings:
    return Settings(
        data_root=str(synthetic_root),
        artifacts=str(tmp_path / "artifacts"),
        n_splits=3,
        features=FeatureConfig(crop_px=96, crop_grid=1, n_levels=5),
    )


def test_features_are_cached_and_reused(settings):
    first = prepare_features(settings, split="train", progress=False)
    cached = list((Path(settings.artifacts) / "features").glob("train_*.csv"))
    assert len(cached) == 1
    second = prepare_features(settings, split="train", progress=False)
    pd.testing.assert_frame_equal(first, second)


def test_changing_feature_settings_invalidates_the_cache(settings):
    prepare_features(settings, split="train", progress=False)
    settings.features = FeatureConfig(crop_px=64, crop_grid=1, n_levels=4)
    prepare_features(settings, split="train", progress=False)
    assert len(list((Path(settings.artifacts) / "features").glob("train_*.csv"))) == 2


def test_cv_writes_its_evidence(settings):
    result = run_cv(settings, model="ridge")
    reports = Path(settings.artifacts) / "cv"
    assert (reports / "summary_ridge.json").exists()
    assert (reports / "oof_ridge.csv").exists()
    assert (reports / "per_sample_ridge.csv").exists()
    assert len(result.fold_scores) == 3
    assert 0.0 <= result.score <= 500.0


def test_submission_is_written_and_valid(settings, synthetic_root):
    path = make_submission(settings, model="ridge")
    assert path.exists()

    template = Path(synthetic_root) / "sample_submission.csv"
    report = validate_submission_file(path, template)
    assert report.ok, report.render()

    written = pd.read_csv(path)
    expected = pd.read_csv(template)
    assert list(written.columns) == list(SUBMISSION_COLUMNS)
    assert written[ID_COLUMN].tolist() == expected[ID_COLUMN].tolist()


def test_settings_round_trip_through_yaml(tmp_path):
    config = tmp_path / "settings.yaml"
    config.write_text(
        "data_root: some/where\nmodel: knn\nfeatures:\n  target_ppm: 8.0\n  n_levels: 5\n",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.data_root == "some/where"
    assert settings.model == "knn"
    assert settings.features.target_ppm == 8.0
    assert load_settings(config, model="gbt").model == "gbt"


def test_cli_runs_cv_submit_and_validate(synthetic_root, tmp_path, capsys):
    artifacts = str(tmp_path / "cli")
    common = ["--data-root", str(synthetic_root), "--artifacts", artifacts]

    assert main(["describe"]) == 0
    assert main(["cv", *common, "--models", "constant", "ridge", "--n-splits", "3"]) == 0
    assert "weighted EMD" in capsys.readouterr().out

    out = str(tmp_path / "submission.csv")
    assert main(["submit", *common, "--model", "ridge", "--out", out]) == 0
    assert main(["validate", out, "--template", str(Path(synthetic_root) / "sample_submission.csv")]) == 0


def test_cli_validate_fails_on_a_broken_file(tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text("sample_id,0.002\nS1,5\n", encoding="utf-8")
    assert main(["validate", str(broken)]) == 1


def test_cli_make_synthetic(tmp_path):
    out = tmp_path / "synth"
    assert main(["make-synthetic", "--out", str(out), "--n-train", "4", "--n-test", "2"]) == 0
    assert (out / "Training_labels_updated.csv").exists()
