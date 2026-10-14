"""Stage 1: whole-cell classification into pyramidal or interneuron.

The predicted type fixes the label set the rest of the pipeline may assign,
since only pyramidal cells carry an apical dendrite. A user who already knows
the type passes it instead, and this stage is skipped.

The classifier reads geometry only: the 55 whole-cell descriptors of
``features.FEATURE_NAMES``, none of which is derived from the SWC type column.
``CellTypeClassifier.load`` refuses a model whose recorded feature names include
a type-derived field, so a model fitted on label distributions cannot be used
here by accident.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._model_cache import load_cached
from ..features import (
    FEATURE_NAMES,
    SWCNode,
    extract_global_features,
    parse_swc,
)

CELL_TYPES = [
    "pyramidal",
    "interneuron",
]

# What label sets each cell type typically uses
CELL_TYPE_LABEL_SETS: dict[str, set[int]] = {
    "pyramidal": {1, 2, 3, 4},       # soma, axon, basal, apical
    "interneuron": {1, 2, 3},         # soma, axon, dendrite (basal only)
}

LEAKY_STAGE1_FEATURES = {
    "frac_type_1",
    "frac_type_2",
    "frac_type_3",
    "frac_type_4",
    "n_unique_types",
}


@dataclass
class CellTypeResult:
    """What Stage 1 decided, and what it decided from."""
    cell_type: str                          # predicted cell type
    confidence: float                       # posterior of the predicted class
    probabilities: dict[str, float]         # per-class probabilities
    label_set: set[int]                     # SWC types the rest of the pipeline may assign
    features: dict[str, float]              # the whole-cell features it read


MODEL_DIR = Path(__file__).parent.parent / "models"
# Seeds 42, 123 and 789 were trained identically; seed 123 is the one the
# paper reports as the deployed labeler, so it is what loads by default.
DEFAULT_MODEL_PATH = MODEL_DIR / "seed123" / "cell_type_classifier.pkl"


class CellTypeClassifier:
    """Wraps a trained sklearn model for cell-type prediction."""

    def __init__(self, model: Any = None, feature_names: list[str] | None = None):
        self.model = model
        self.feature_names = feature_names or list(FEATURE_NAMES)
        self._classes: list[str] = list(CELL_TYPES)

    @classmethod
    def load(cls, model_path: str | Path = DEFAULT_MODEL_PATH) -> "CellTypeClassifier":
        """Load a trained model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No Stage 1 model at {model_path}. Train one with "
                "`python -m pipeline.train_pipeline`, or give the cell type yourself "
                "(`run_pipeline_on_nodes(..., override_cell_type=...)`), which skips Stage 1."
            )
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"Stage 1 model at {model_path} could not be read ({type(exc).__name__}: {exc}). "
                "Models are saved with the package versions in requirements.txt; another "
                "scikit-learn or xgboost version may not read them."
            ) from exc
        obj = cls()
        obj.model = data["model"]
        obj._classes = data.get("classes", list(CELL_TYPES))
        obj.feature_names = data.get("feature_names", list(FEATURE_NAMES))
        if any(name in LEAKY_STAGE1_FEATURES for name in obj.feature_names):
            raise ValueError(
                f"Leaky Stage 1 model at {model_path}: feature_names contain SWC type-derived fields. Retrain via pipeline.stage1.train."
            )
        if obj.feature_names != list(FEATURE_NAMES):
            raise ValueError(
                f"Stage 1 model at {model_path} has stale feature schema. Retrain via pipeline.stage1.train."
            )
        return obj

    def save(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """Save the trained model to disk."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "classes": self._classes,
                "feature_names": self.feature_names,
            }, f)

    def predict(self, feature_vector: np.ndarray) -> tuple[str, dict[str, float]]:
        """Predict cell type from a feature vector.

        Returns (cell_type, probability_dict).
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load() first.")

        X = feature_vector.reshape(1, -1)
        probs = self.model.predict_proba(X)[0]
        prob_dict = {c: float(p) for c, p in zip(self._classes, probs)}
        best = self._classes[int(np.argmax(probs))]
        return best, prob_dict


def load_classifier(model_path: str | Path | None = None) -> CellTypeClassifier:
    """The Stage 1 classifier at ``model_path`` (default: the shipped one), read
    once per file state."""
    return load_cached(Path(model_path) if model_path else DEFAULT_MODEL_PATH,
                       CellTypeClassifier.load)


def detect_cell_type(
    swc_path: str | Path,
    model_path: str | Path | None = None,
) -> CellTypeResult:
    """Run Stage 1 cell-type detection on an SWC file."""
    return detect_cell_type_from_nodes(parse_swc(swc_path), model_path)


def detect_cell_type_from_nodes(
    nodes: list[SWCNode],
    model_path: str | Path | None = None,
) -> CellTypeResult:
    """Run Stage 1 on pre-parsed nodes (avoids re-reading file).

    The trained model decides. Anything that stops it deciding raises: a missing
    or unreadable model, or a file that parsed to nothing. A cell type guessed
    some other way would look the same to the caller, so the failure is reported
    rather than hidden.
    """
    if not nodes:
        raise ValueError(
            "no nodes to classify: the file parsed to zero nodes. Check that it is "
            "an SWC file, or pass the cell type yourself to skip Stage 1.")

    features = extract_global_features(nodes)
    feature_vec = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)

    cell_type, probs = load_classifier(model_path).predict(feature_vec)

    return CellTypeResult(
        cell_type=cell_type,
        confidence=probs.get(cell_type, 0.0),
        probabilities=probs,
        label_set=set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3})),
        features=features,
    )
