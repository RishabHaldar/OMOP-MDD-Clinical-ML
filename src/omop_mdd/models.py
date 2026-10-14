"""Model pipelines. All preprocessing is fitted inside the Pipeline."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from omop_mdd import config
from omop_mdd.features import AllowlistGuard

BINARY_PREDICTORS = ["is_male"]


def _imputer() -> ColumnTransformer:
    """Training-only imputation, identical for both models.

    * ``is_male``: most frequent training value, so imputed values stay 0 or 1
      (ties resolve to 0, SimpleImputer's documented behaviour).
    * every other predictor: training median.

    Fitted on training rows only, without labels (SimpleImputer ignores y).
    Transformers are listed so the output keeps the allowlist column order.
    """
    predictors = config.predictors()
    binary_at = predictors.index(BINARY_PREDICTORS[0])
    before = predictors[:binary_at]
    after = predictors[binary_at + 1 :]

    def median() -> SimpleImputer:
        return SimpleImputer(strategy="median", keep_empty_features=True)

    return ColumnTransformer(
        [
            ("continuous_before", median(), before),
            (
                "binary",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                BINARY_PREDICTORS,
            ),
            ("continuous_after", median(), after),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def imputation_values(pipe: Pipeline) -> pd.Series:
    """Fitted imputation value per predictor, in allowlist order."""
    ct = pipe.named_steps["impute"]
    values = {}
    for _, imputer, columns in ct.transformers_:
        if imputer == "drop":
            continue
        values.update(dict(zip(columns, imputer.statistics_)))
    return pd.Series(values).loc[config.predictors()]


def logistic_regression(seed: int) -> Pipeline:
    """Baseline: shared imputation, standardization, L2 logistic regression."""
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
