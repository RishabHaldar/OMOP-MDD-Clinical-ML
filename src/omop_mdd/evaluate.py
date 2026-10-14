"""Patient-level split, training-only cross-validation, single held-out test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ids_train: pd.Series
    ids_test: pd.Series


def split_patients(
    X: pd.DataFrame, y: pd.Series, ids: pd.Series, test_size: float, seed: int
) -> Split:
    """Stratified patient-level split. One row per patient, so no patient can
    appear in both sets."""
    if ids.duplicated().any():
        raise ValueError("person_id must be unique before splitting.")
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed
    )
    return Split(
        X_train=X.iloc[idx_train].reset_index(drop=True),
        X_test=X.iloc[idx_test].reset_index(drop=True),
        y_train=y.iloc[idx_train].reset_index(drop=True),
        y_test=y.iloc[idx_test].reset_index(drop=True),
        ids_train=ids.iloc[idx_train].reset_index(drop=True),
        ids_test=ids.iloc[idx_test].reset_index(drop=True),
    )


def cross_validate_training(
    pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, folds: int, seed: int
) -> dict:
    """Stratified k-fold CV on the training set only. The whole Pipeline,
    including the imputer and scaler, is refitted inside every fold."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = cross_validate(
        clone(pipe), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
    )
    scores = result["test_score"]
    return {
        "auroc_folds": [float(s) for s in scores],
        "auroc_mean": float(np.mean(scores)),
        "auroc_sd": float(np.std(scores, ddof=1)),
    }


def bootstrap_auroc_ci(
    y_true: np.ndarray, scores: np.ndarray, n_resamples: int, seed: int
) -> tuple[float, float]:
    """Percentile 95% CI for AUROC by resampling test patients."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(roc_auc_score(y_true[idx], scores[idx]))
    lower, upper = np.percentile(stats, [2.5, 97.5])
    return float(lower), float(upper)


def evaluate_test(
    fitted: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    n_resamples: int,
    seed: int,
) -> tuple[dict, np.ndarray]:
    """Single held-out evaluation. Returns (metrics, predicted probabilities)."""
    prob = fitted.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    lower, upper = bootstrap_auroc_ci(y_test, prob, n_resamples, seed)
    metrics = {
        "n_test": int(len(y_test)),
        "auroc": float(roc_auc_score(y_test, prob)),
        "auroc_95ci_bootstrap": [lower, upper],
        "threshold": float(threshold),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    return metrics, prob


def importance(
    fitted: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, seed: int
) -> pd.DataFrame:
    """Model-specific importance plus permutation importance on the test set."""
    names = list(X_test.columns)
    # Coefficients and gains are indexed by the transformed columns; check they
    # are the same 18 predictors in the same order before labelling them.
    transformed = list(fitted[:-1].get_feature_names_out())
    if transformed != names:
        raise ValueError(f"Transformed feature order differs: {transformed}")
    model = fitted.named_steps["model"]
    out = pd.DataFrame({"feature": names})
    if hasattr(model, "coef_"):
        # Coefficients on standardized predictors (log-odds per 1 SD).
        out["coefficient_std"] = model.coef_[0]
    if hasattr(model, "booster_"):
        out["gain"] = model.booster_.feature_importance(importance_type="gain")
    perm = permutation_importance(
        fitted,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=50,
        random_state=seed,
        n_jobs=1,
    )
    out["permutation_auroc_drop_mean"] = perm.importances_mean
    out["permutation_auroc_drop_sd"] = perm.importances_std
    return out
