from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soilgsd.constants import ID_COLUMN, TARGET_COLUMNS
from soilgsd.curves import is_valid, project_valid
from soilgsd.evaluate import cross_validate, derive_groups
from soilgsd.metric import weighted_emd
from soilgsd.models import MODEL_REGISTRY, build_model

MODEL_NAMES = sorted(MODEL_REGISTRY) + ["blend:ridge+knn"]


@pytest.fixture
def toy():
    """Features that carry real signal about the curve, plus pure noise columns."""
    rng = np.random.default_rng(0)
    n = 80
    coarseness = rng.uniform(-1.0, 1.0, size=n)
    supports = np.linspace(-1.5, 1.5, 11)
    curves = 100.0 / (1.0 + np.exp(-(supports[None, :] - coarseness[:, None] * 1.5) * 2.2))
    curves = project_valid(curves + rng.normal(scale=1.0, size=curves.shape))
    features = pd.DataFrame(
        {
            "signal": coarseness + rng.normal(scale=0.05, size=n),
            "noise_a": rng.normal(size=n),
            "noise_b": rng.normal(size=n),
        }
    )
    return features, pd.DataFrame(curves, columns=list(TARGET_COLUMNS))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_predicts_submittable_curves(name, toy):
    features, curves = toy
    predicted = build_model(name).fit(features, curves).predict(features)
    assert predicted.shape == curves.shape
    assert is_valid(predicted).all(), name


@pytest.mark.parametrize("name", [n for n in MODEL_NAMES if n != "constant"])
def test_feature_models_beat_the_constant_baseline(name, toy):
    features, curves = toy
    baseline = build_model("constant").fit(features, curves).predict(features)
    predicted = build_model(name).fit(features, curves).predict(features)
    assert weighted_emd(curves, predicted) < weighted_emd(curves, baseline), name


def test_constant_model_predicts_the_training_median(toy):
    _, curves = toy
    model = build_model("constant").fit(None, curves)
    predicted = model.predict(range(5))
    assert predicted.shape == (5, 11)
    assert (predicted == predicted[0]).all()


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("does-not-exist")


def test_blend_weights_must_be_usable():
    from soilgsd.models import BlendCurve

    with pytest.raises(ValueError, match="same length"):
        BlendCurve([build_model("constant")], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="at least one member"):
        BlendCurve([])


def test_cross_validation_reports_per_fold_scores(toy):
    features, curves = toy
    features = features.copy()
    features.insert(0, ID_COLUMN, [f"s{i:03d}" for i in range(len(features))])
    labels = curves.copy()
    labels.insert(0, ID_COLUMN, features[ID_COLUMN].to_numpy())

    result = cross_validate(lambda: build_model("ridge"), features, labels, n_splits=4)
    assert len(result.fold_scores) == 4
    assert result.score == pytest.approx(np.mean(result.per_sample["emd"]))
    assert is_valid(result.oof_predictions[list(TARGET_COLUMNS)].to_numpy()).all()
    assert len(result.per_support_mae) == 11


def test_grouped_cv_keeps_a_group_whole(toy):
    features, curves = toy
    ids = [f"SOIL{i // 4:02d}_rep{i % 4}" for i in range(len(features))]
    features = features.copy()
    features.insert(0, ID_COLUMN, ids)
    labels = curves.copy()
    labels.insert(0, ID_COLUMN, ids)

    groups = derive_groups(pd.Series(ids), r"^([A-Za-z]+[0-9]+)")
    assert groups.nunique() == len(ids) // 4

    result = cross_validate(
        lambda: build_model("knn"), features, labels, n_splits=3, group_pattern=r"^([A-Za-z]+[0-9]+)"
    )
    assert len(result.fold_scores) == 3


def test_derive_groups_defaults_to_the_sample_id():
    ids = pd.Series(["a", "b", "c"])
    assert list(derive_groups(ids)) == ["a", "b", "c"]
