"""Predictor allowlist enforcement (leakage guard)."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from omop_mdd import config


class LeakageError(ValueError):
    """Raised when a non-allowlisted column would reach a model."""


def check_allowlist(columns: list[str]) -> None:
    """Raise unless ``columns`` is exactly the predictor allowlist, in order."""
    allowed = config.predictors()
    forbidden = set(config.features()["non_predictors"])
    named = [c for c in columns if c in forbidden]
    if named:
        raise LeakageError(f"Non-predictor column(s) present: {named}")
    extra = [c for c in columns if c not in allowed]
    if extra:
        raise LeakageError(f"Column(s) not in the predictor allowlist: {extra}")
    missing = [c for c in allowed if c not in columns]
    if missing:
        raise LeakageError(f"Allowlisted predictor(s) missing: {missing}")
    if list(columns) != allowed:
        raise LeakageError("Predictor columns are not in allowlist order.")


def select_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the allowlisted predictor columns from ``df``."""
    allowed = config.predictors()
    missing = [c for c in allowed if c not in df.columns]
    if missing:
        raise LeakageError(f"Allowlisted predictor(s) missing: {missing}")
    X = df.loc[:, allowed].astype(float)
    check_allowlist(list(X.columns))
    return X


class AllowlistGuard(BaseEstimator, TransformerMixin):
    """First pipeline step: refuses any input that is not the exact allowlist.

    Placing the check inside the Pipeline means it runs on every fit, every
    cross-validation fold and every prediction, not just once at data load.
    """

    def fit(self, X, y=None):
        check_allowlist(list(getattr(X, "columns", [])))
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        check_allowlist(list(getattr(X, "columns", [])))
        return X

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names_in_)
