"""GPU-accelerated drop-in replacements for the sklearn ensemble classifiers
used in Stage 2 training.

Each class below mirrors the keyword-argument signature of its sklearn
counterpart so the call site in ``pipeline/stage2/train_stage2.py`` can swap in
via a one-line `from sklearn.ensemble import ...` -> `from
pipeline._xgb_classifiers import ...` change, with NO modification to the
constructor calls themselves. Stage 1 does not use these: it constructs an
``xgb.XGBClassifier`` directly.

Algorithmic mapping (kept as close as possible to sklearn behavior):

    sklearn.RandomForestClassifier
        -> XGB "random-forest mode": num_parallel_tree=n_estimators,
           n_estimators=1, subsample=0.8, colsample_bynode=0.8,
           learning_rate=1.0. Structurally a bagged ensemble of trees.

    sklearn.GradientBoostingClassifier
        -> XGB vanilla boosting (n_estimators rounds of additive trees).

Implementation note
-------------------
We compose (have-a XGBClassifier) rather than inherit (is-a) so the public
constructor signature can use sklearn parameter names without colliding
with the parent class's own parameters of the same name. The wrappers
inherit sklearn.base.BaseEstimator / ClassifierMixin so they remain fully
sklearn-API-compatible: ``fit``, ``predict``, ``predict_proba``,
``classes_``, plus ``get_params`` / ``set_params`` / ``sklearn.base.clone``
required by VotingClassifier and CalibratedClassifierCV.

Two parameter names differ from sklearn's on purpose. ``min_child_weight`` is
XGBoost's own parameter and is NOT sklearn's ``min_samples_leaf``: sklearn
counts samples in a leaf, XGBoost sums their hessians, which for a classifier
are at most 0.25 apiece, so the same number means a much larger leaf here. It
keeps XGBoost's name so the value is read as XGBoost defines it, and it is the
quantity Supplementary Section S2.1 reports. ``device`` defaults to ``'cuda'``, which is how
the released models were fitted; pass ``device='cpu'`` to train without an
NVIDIA GPU.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def _label_encode(y):
    """Map arbitrary integer labels (e.g. {2,3,4}) to {0..N-1}.

    XGBoost 3.x rejects non-contiguous labels. We encode at fit time and
    decode at predict time so the wrapper behaves identically to sklearn
    classifiers from the caller's point of view (``classes_`` and the
    return value of ``predict`` are in the original label space).
    """
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    lookup = {c: i for i, c in enumerate(classes.tolist())}
    enc = np.array([lookup[v] for v in y_arr.tolist()], dtype=np.int64)
    return enc, classes


def _switch_inference_to_cpu(model: xgb.XGBClassifier) -> None:
    """Flip the booster's runtime device to CPU after training on GPU.

    GPU launch overhead (~1-5 ms per call) dominates for the small per-cell
    predict batches the pipeline issues (~2000 branches at a time). Training
    keeps the GPU win on the big batched fit; inference switches to CPU
    where small predicts are essentially free. Same weights, same
    predictions, just faster.
    """
    try:
        model.get_booster().set_param({"device": "cpu"})
    except Exception:
        # If anything goes wrong, leave the model as-is; predictions will
        # still be correct, just slower.
        pass


# ---------------------------------------------------------------------------
# Random Forest (XGBoost "random forest mode")
# ---------------------------------------------------------------------------
class XGBRandomForestClassifier(ClassifierMixin, BaseEstimator):
    """GPU drop-in for sklearn.ensemble.RandomForestClassifier.

    Algorithm: XGBoost RF mode. One boosting round in which
    ``num_parallel_tree`` independent trees are fit in parallel (bagging),
    with ``subsample`` and ``colsample_bynode`` introducing per-tree
    randomness. Structurally a bagged ensemble of trees like sklearn RF.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_child_weight: int = 1,
        class_weight: Any = None,
        random_state: int = 42,
        n_jobs: int | None = None,   # ignored on GPU
        device: str = "cuda",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device

    def _make_xgb(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=1,                                       # 1 boosting round
            num_parallel_tree=self.n_estimators,                  # N trees in parallel
            max_depth=0 if self.max_depth is None else self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=0.8,
            colsample_bynode=0.8,
            learning_rate=1.0,
            device=self.device,
            tree_method="hist",
            random_state=self.random_state,
            verbosity=0,
        )

    def fit(self, X, y, sample_weight=None):
        y_enc, classes = _label_encode(y)
        if self.class_weight == "balanced" and sample_weight is None:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight(class_weight="balanced", y=np.asarray(y))
        self._model_ = self._make_xgb()
        self._model_.fit(X, y_enc, sample_weight=sample_weight)
        _switch_inference_to_cpu(self._model_)
        self.classes_ = classes
        return self

    def predict(self, X):
        enc = self._model_.predict(X)
        return self.classes_[np.asarray(enc).astype(int)]

    def predict_proba(self, X):
        return self._model_.predict_proba(X)


# ---------------------------------------------------------------------------
# Vanilla gradient boosting
# ---------------------------------------------------------------------------
class XGBGradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    """GPU drop-in for sklearn.ensemble.GradientBoostingClassifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        random_state: int = 42,
        device: str = "cuda",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.device = device

    def fit(self, X, y, sample_weight=None):
        y_enc, classes = _label_encode(y)
        self._model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            device=self.device,
            tree_method="hist",
            random_state=self.random_state,
            verbosity=0,
        )
        self._model_.fit(X, y_enc, sample_weight=sample_weight)
        _switch_inference_to_cpu(self._model_)
        self.classes_ = classes
        return self

    def predict(self, X):
        enc = self._model_.predict(X)
        return self.classes_[np.asarray(enc).astype(int)]

    def predict_proba(self, X):
        return self._model_.predict_proba(X)

