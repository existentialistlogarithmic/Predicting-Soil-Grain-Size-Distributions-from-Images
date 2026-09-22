"""Models that map sample-level features onto valid cumulative curves.

Every model here returns curves that already satisfy the submission
constraints, so a prediction is never one forgotten projection away from being
rejected.

The metric is an absolute-error metric (see :mod:`soilgsd.metric`), so the
models are built around L1 rather than L2 wherever the choice exists: the
constant baseline is a median, the gradient-boosting model uses the
absolute-error loss, and the neighbour model pools by median.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import LOG_SUPPORTS, N_SUPPORTS, TARGET_COLUMNS
from .curves import median_curve, project_valid

__all__ = [
    "CurveModel",
    "ConstantCurve",
    "RidgeCurve",
    "GradientBoostedCurve",
    "NeighbourCurve",
    "ScalarBottleneckCurve",
    "BlendCurve",
    "build_model",
    "MODEL_REGISTRY",
]


def _as_matrix(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(features, pd.DataFrame):
        features = features.to_numpy(dtype=np.float64)
    return np.asarray(features, dtype=np.float64)


def _as_targets(targets: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(targets, pd.DataFrame):
        targets = targets.reindex(columns=list(TARGET_COLUMNS)).to_numpy(dtype=np.float64)
    array = np.asarray(targets, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != N_SUPPORTS:
        raise ValueError(f"expected targets of shape (n, {N_SUPPORTS}), got {array.shape}")
    return array


class CurveModel:
    """Base class: ``fit`` on sample features and curves, ``predict`` valid curves."""

    name = "base"

    def fit(self, features: pd.DataFrame | np.ndarray, targets: pd.DataFrame | np.ndarray) -> "CurveModel":
        raise NotImplementedError

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(self, features, targets, test_features) -> np.ndarray:
        return self.fit(features, targets).predict(test_features)


class ConstantCurve(CurveModel):
    """Predict one curve for every sample: the training median.

    Under an absolute-error metric this is the optimal feature-blind
    prediction, and it is the number every image model has to beat before it is
    worth anything.
    """

    name = "constant"

    def __init__(self) -> None:
        self.curve_: np.ndarray | None = None

    def fit(self, features, targets) -> "ConstantCurve":
        self.curve_ = median_curve(_as_targets(targets))
        return self

    def predict(self, features) -> np.ndarray:
        if self.curve_ is None:
            raise RuntimeError("call fit before predict")
        n_rows = len(features)
        return np.tile(self.curve_, (n_rows, 1))


@dataclass
class _Standardiser:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, matrix: np.ndarray) -> "_Standardiser":
        self.mean_ = np.nanmean(matrix, axis=0)
        scale = np.nanstd(matrix, axis=0)
        # A constant column carries no information; leaving its scale at 1
        # keeps it at zero after centring instead of exploding it.
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("call fit before transform")
        out = (np.nan_to_num(matrix, nan=np.nan) - self.mean_) / self.scale_
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class RidgeCurve(CurveModel):
    """Ridge regression onto the ten free supports, then projection.

    Cheap, stable on small sample counts, and a good sanity check that the
    features carry signal at all.  ``alpha`` is chosen inside ``fit`` by
    leave-one-out generalised cross-validation, so no outer tuning leaks.
    """

    name = "ridge"

    def __init__(self, alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)) -> None:
        self.alphas = alphas
        self.scaler_ = _Standardiser()
        self.model_ = None

    def fit(self, features, targets) -> "RidgeCurve":
        from sklearn.linear_model import RidgeCV

        matrix = self.scaler_.fit(_as_matrix(features)).transform(_as_matrix(features))
        self.model_ = RidgeCV(alphas=self.alphas).fit(matrix, _as_targets(targets)[:, :-1])
        return self

    def predict(self, features) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit before predict")
        matrix = self.scaler_.transform(_as_matrix(features))
        free = self.model_.predict(matrix)
        full = np.column_stack([free, np.full(len(free), 100.0)])
        return project_valid(full)


class GradientBoostedCurve(CurveModel):
    """One absolute-error gradient-boosting regressor per free support.

    Fitting each support independently ignores the coupling between them, which
    the projection step then repairs.  The alternative — predicting bin masses
    on the simplex — keeps the coupling but forces a squared-error objective,
    which is the wrong loss here; per-support L1 wins in practice because the
    metric decomposes over supports in exactly the same way.
    """

    name = "gbt"

    def __init__(
        self,
        *,
        max_iter: int = 300,
        learning_rate: float = 0.06,
        max_leaf_nodes: int = 15,
        min_samples_leaf: int = 8,
        l2_regularization: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.params = dict(
            loss="absolute_error",
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=False,
            random_state=random_state,
        )
        self.models_: list = []

    def fit(self, features, targets) -> "GradientBoostedCurve":
        from sklearn.ensemble import HistGradientBoostingRegressor

        matrix = _as_matrix(features)
        curves = _as_targets(targets)
        self.models_ = [
            HistGradientBoostingRegressor(**self.params).fit(matrix, curves[:, support])
            for support in range(N_SUPPORTS - 1)
        ]
        return self

    def predict(self, features) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("call fit before predict")
        matrix = _as_matrix(features)
        free = np.column_stack([model.predict(matrix) for model in self.models_])
        full = np.column_stack([free, np.full(len(free), 100.0)])
        return project_valid(full)


class NeighbourCurve(CurveModel):
    """Median of the ``k`` nearest training curves in standardised feature space.

    Because a pointwise median of valid curves is itself valid, this model can
    never produce an unphysical prediction, and it degrades gracefully to the
    constant baseline as ``k`` grows.

    ``per_domain`` standardises a batch of predictions using that batch's own
    mean and standard deviation rather than the training set's.  **It is off by
    default because it costs 17 EMD on the leaderboard: 85.06 with it, 68.07
    without.**

    It is kept because the reasoning for it was not silly and the failure is
    instructive.  The training and test feature clouds really are offset - test
    samples sat 1.25x the training spread away from their nearest training
    neighbours, and switching this on cut that to 0.83x and visibly restored
    prediction variety.  Both of those diagnostics improved.  The score got
    much worse anyway, because the features are already calibrated to
    millimetres through the camera scale, and rescaling them to a batch's own
    spread discards exactly that calibration.  The offset was the price of
    keeping a physically meaningful axis, not a fault to be corrected.

    When on, it assumes the batch handed to ``predict`` is a whole domain, so
    batches smaller than ``min_domain_rows`` fall back to training statistics.
    """

    name = "knn"

    def __init__(
        self,
        n_neighbours: int = 7,
        *,
        per_domain: bool = False,
        min_domain_rows: int = 5,
    ) -> None:
        self.n_neighbours = n_neighbours
        self.per_domain = per_domain
        self.min_domain_rows = min_domain_rows
        self.scaler_ = _Standardiser()
        self.train_matrix_: np.ndarray | None = None
        self.train_curves_: np.ndarray | None = None

    def fit(self, features, targets) -> "NeighbourCurve":
        matrix = _as_matrix(features)
        # The training cloud is always centred on its own statistics; per_domain
        # only changes how a prediction batch is placed into that same frame.
        self.train_matrix_ = self.scaler_.fit(matrix).transform(matrix)
        self.train_curves_ = _as_targets(targets)
        return self

    def _project(self, features) -> np.ndarray:
        matrix = _as_matrix(features)
        if self.per_domain and len(matrix) >= self.min_domain_rows:
            local = _Standardiser().fit(matrix)
            return local.transform(matrix)
        return self.scaler_.transform(matrix)

    def predict(self, features) -> np.ndarray:
        if self.train_matrix_ is None or self.train_curves_ is None:
            raise RuntimeError("call fit before predict")
        matrix = self._project(features)
        k = int(min(self.n_neighbours, len(self.train_matrix_)))
        distances = (
            (matrix**2).sum(axis=1)[:, None]
            - 2.0 * matrix @ self.train_matrix_.T
            + (self.train_matrix_**2).sum(axis=1)[None, :]
        )
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        pooled = np.median(self.train_curves_[nearest], axis=1)
        return project_valid(pooled)


class BlendCurve(CurveModel):
    """Weighted pointwise blend of several fitted models.

    A convex blend of valid curves is valid, so the projection at the end only
    guards against floating-point drift.
    """

    name = "blend"

    def __init__(self, members: list[CurveModel], weights: list[float] | None = None) -> None:
        if not members:
            raise ValueError("a blend needs at least one member")
        self.members = members
        if weights is None:
            weights = [1.0 / len(members)] * len(members)
        if len(weights) != len(members):
            raise ValueError("weights and members must have the same length")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("blend weights must sum to a positive number")
        self.weights = [float(weight) / total for weight in weights]

    def fit(self, features, targets) -> "BlendCurve":
        for member in self.members:
            member.fit(features, targets)
        return self

    def predict(self, features) -> np.ndarray:
        stacked = np.stack([member.predict(features) for member in self.members])
        weights = np.asarray(self.weights).reshape(-1, 1, 1)
        return project_valid((stacked * weights).sum(axis=0))


def _log_d50(curves: np.ndarray, percentile: float = 50.0) -> np.ndarray:
    """log10 of the diameter at which each curve passes ``percentile``."""
    out = np.empty(len(curves))
    for row, curve in enumerate(curves):
        rising = np.maximum.accumulate(curve)
        out[row] = (
            LOG_SUPPORTS[0]
            if rising[0] >= percentile
            else float(np.interp(percentile, rising, LOG_SUPPORTS))
        )
    return out


class ScalarBottleneckCurve(CurveModel):
    """Predict one number - the median grain diameter - then look up a curve.

    Twenty-two texture features over twenty-four training samples is a lot of
    freedom, and it showed: the neighbour model's test predictions sat outside
    the training feature cloud and barely varied.  Forcing the image through a
    single scalar removes almost all of that freedom.  Grain size is what the
    photograph actually measures, the strongest individual feature correlates
    with log10(d50) at r = 0.86, and a one-dimensional space is far harder to
    land outside of than a twenty-two dimensional one.

    The curve comes from pooling the training soils whose d50 is closest to the
    prediction, which keeps real curve shapes rather than assuming a lognormal.

    Features are standardised against the **training** distribution, not within
    the batch being predicted.  Normalising within the batch looks like sensible
    domain adaptation and is not: the features are already calibrated to
    millimetres via the camera scale, so rescaling them to the batch's own
    spread throws that calibration away.  Doing exactly that to the neighbour
    model cost 17 EMD on the leaderboard (85.06 with it, 68.07 without), so
    this model does not repeat it.

    Feature selection happens inside ``fit``, on the training fold only, so
    cross-validation stays honest.
    """

    name = "d50"

    def __init__(self, n_features: int = 5, n_neighbours: int = 7) -> None:
        self.n_features = n_features
        self.n_neighbours = n_neighbours
        self.scaler_ = _Standardiser()
        self.selected_: np.ndarray | None = None
        self.coefficients_: np.ndarray | None = None
        self.train_d50_: np.ndarray | None = None
        self.train_curves_: np.ndarray | None = None

    def fit(self, features, targets) -> "ScalarBottleneckCurve":
        raw = _as_matrix(features)
        matrix = self.scaler_.fit(raw).transform(raw)
        curves = _as_targets(targets)
        target_d50 = _log_d50(curves)

        spread = matrix.std(axis=0)
        correlation = np.zeros(matrix.shape[1])
        usable = spread > 1e-12
        if usable.any():
            centred = matrix[:, usable] - matrix[:, usable].mean(axis=0)
            centred_target = target_d50 - target_d50.mean()
            denominator = np.sqrt((centred**2).sum(axis=0) * (centred_target**2).sum())
            correlation[usable] = np.abs(centred.T @ centred_target) / np.maximum(denominator, 1e-12)

        count = int(min(self.n_features, matrix.shape[1]))
        self.selected_ = np.argsort(correlation)[::-1][:count]
        design = np.column_stack([matrix[:, self.selected_], np.ones(len(matrix))])
        self.coefficients_ = np.linalg.lstsq(design, target_d50, rcond=None)[0]
        self.train_d50_ = target_d50
        self.train_curves_ = curves
        return self

    def predict(self, features) -> np.ndarray:
        if self.coefficients_ is None or self.train_curves_ is None:
            raise RuntimeError("call fit before predict")
        matrix = self.scaler_.transform(_as_matrix(features))
        design = np.column_stack([matrix[:, self.selected_], np.ones(len(matrix))])
        predicted = design @ self.coefficients_
        # Never claim a grain size the training set gives no evidence for.
        predicted = np.clip(predicted, self.train_d50_.min(), self.train_d50_.max())

        k = int(min(self.n_neighbours, len(self.train_curves_)))
        pooled = np.empty((len(predicted), N_SUPPORTS))
        for row, value in enumerate(predicted):
            nearest = np.argsort(np.abs(self.train_d50_ - value))[:k]
            pooled[row] = np.median(self.train_curves_[nearest], axis=0)
        return project_valid(pooled)


MODEL_REGISTRY: dict[str, type[CurveModel]] = {
    ScalarBottleneckCurve.name: ScalarBottleneckCurve,
    ConstantCurve.name: ConstantCurve,
    RidgeCurve.name: RidgeCurve,
    GradientBoostedCurve.name: GradientBoostedCurve,
    NeighbourCurve.name: NeighbourCurve,
}


def build_model(name: str, **kwargs) -> CurveModel:
    """Instantiate a registered model by name, or a ``blend:a+b`` of several."""
    if name.startswith("blend:"):
        parts = [part.strip() for part in name.split(":", 1)[1].split("+") if part.strip()]
        return BlendCurve([build_model(part) for part in parts])
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)} or 'blend:a+b'"
        ) from None
    return factory(**kwargs)
