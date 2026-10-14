"""Stage 1 training: whole-cell cell-type classifier (pyramidal vs interneuron).

A single XGBoost classifier over the whole-cell feature vector, with the minority
class oversampled to the majority count and the number of boosting rounds chosen
by early stopping on a held-out slice.

Entry point:
    train_stage1(train_files, model_path, workers=1)
        train_files: {cell_type: [swc paths]}, from the training split alone;
                     passing test cells here would leak them into the model
        model_path:  where to write cell_type_classifier.pkl
        workers:     processes for feature extraction (>1 parallelises it)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .cell_type_detector import CELL_TYPES, CellTypeClassifier
from ..features import FEATURE_NAMES, extract_feature_vector, parse_swc


def _extract_feature_row(item: tuple[str, int]) -> tuple[np.ndarray, int] | None:
    """Feature-extraction unit of work: (swc path, class index) -> (features, class).

    Module-level so it is picklable for parallel extraction. Returns None for
    files that fail to parse, matching the serial skip behaviour.
    """
    path_str, class_index = item
    try:
        nodes = parse_swc(path_str)
        if not nodes:
            return None
        return extract_feature_vector(nodes), class_index
    except Exception:
        return None


def train_stage1(train_files: dict[str, list[Path]], model_path: Path, workers: int = 1,
                 device: str = "cuda") -> None:
    """Train Stage 1 on the training split alone.

    The random state is 42 whatever corpus seed is being built. A seed selects
    which files are in the split, and that is the whole of what separates the
    three released models; the fit itself is not re-randomised. Supplementary Section S2.1 of
    the paper reports it this way. ``device`` is where the fit runs: ``cuda``, as
    the released models were fitted, or ``cpu`` on a machine without an NVIDIA
    GPU. The saved model predicts on the CPU either way.
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    class_names = sorted(train_files.keys())
    tasks = [(str(f), ci) for ci, ct in enumerate(class_names) for f in train_files[ct]]
    # Feature extraction is embarrassingly parallel and order-preserving, so the
    # resulting matrix (and the trained model) is identical to a serial pass.
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_extract_feature_row, tasks, chunksize=8))
    else:
        rows = [_extract_feature_row(t) for t in tasks]
    rows = [r for r in rows if r is not None]

    X = np.stack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])

    # class_names sorted alphabetically -> 0=interneuron, 1=pyramidal.
    # Match the majority class count by bootstrap-sampling the minority, so the
    # model trains on balanced data without class-weight tricks.
    rng = np.random.default_rng(42)
    unique, counts = np.unique(y, return_counts=True)
    n_max = int(counts.max())
    new_X_parts, new_y_parts = [X], [y]
    for cls, cnt in zip(unique.tolist(), counts.tolist()):
        need = n_max - cnt
        if need <= 0:
            continue
        class_idx = np.flatnonzero(y == cls)
        pick = rng.choice(class_idx, size=need, replace=True)
        new_X_parts.append(X[pick])
        new_y_parts.append(y[pick])
    X = np.concatenate(new_X_parts, axis=0)
    y = np.concatenate(new_y_parts, axis=0)
    print(f"  Oversampled training to balanced: {len(y)} samples "
          f"({(y == 0).sum()} interneuron + {(y == 1).sum()} pyramidal)")

    # The oversampling above precedes this split, so a duplicated minority cell
    # can land on both sides of it and the stopping error reads slightly better
    # than it is. That decides the round count and nothing else: the reported
    # accuracy is measured on held-out files this model never saw.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42,
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)
    clf = xgb.XGBClassifier(
        n_estimators=2000, max_depth=8, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=2, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="error",
        tree_method="hist", device=device,
        random_state=42, n_jobs=4,
        early_stopping_rounds=50,
    )
    clf.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
    print(f"  XGB stopped at iteration {clf.best_iteration} of 2000 "
          f"(best val error={clf.best_score:.4f})")
    clf.get_booster().set_param({"device": "cpu"})  # CPU inference
    pipeline = Pipeline([("scaler", scaler), ("clf", clf)])

    full_classes = list(class_names)
    for ct in CELL_TYPES:
        if ct not in full_classes:
            full_classes.append(ct)

    classifier = CellTypeClassifier()
    classifier.model = pipeline
    classifier._classes = full_classes
    classifier.feature_names = list(FEATURE_NAMES)
    classifier.save(model_path)
