"""Model pipelines. All preprocessing is fitted inside the Pipeline."""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from omop_mdd.features import AllowlistGuard


def _imputer() -> SimpleImputer:
    # One median imputer per pipeline, fitted on training rows only, without
    # labels (SimpleImputer ignores y).
    return SimpleImputer(strategy="median", keep_empty_features=True)


def logistic_regression(seed: int) -> Pipeline:
    """Baseline: median imputation, standardization, L2 logistic regression."""
    pipe = Pipeline(
        [
            ("guard", AllowlistGuard()),
            ("impute", _imputer()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=seed)),
        ]
    )
    return pipe.set_output(transform="pandas")


def lightgbm(seed: int) -> Pipeline:
    """LightGBM with fixed, conservative settings (no tuning)."""
    pipe = Pipeline(
        [
            ("guard", AllowlistGuard()),
            ("impute", _imputer()),
            (
                "model",
                LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=10,
                    random_state=seed,
                    n_jobs=1,
                    deterministic=True,
                    force_row_wise=True,
                    verbose=-1,
                ),
            ),
        ]
    )
    return pipe.set_output(transform="pandas")


MODELS = {
    "logistic_regression": logistic_regression,
    "lightgbm": lightgbm,
}
