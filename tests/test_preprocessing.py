"""Preprocessing is fitted on training data only, without labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_validate

from omop_mdd import config
from omop_mdd.evaluate import split_patients
from omop_mdd.models import BINARY_PREDICTORS, MODELS, imputation_values

EVAL = config.analysis()["evaluation"]
CONTINUOUS = [c for c in config.predictors() if c not in BINARY_PREDICTORS]


@pytest.fixture(scope="module")
def split(prepared):
    X, y, ids = prepared
    return split_patients(X, y, ids, EVAL["test_size"], seed=0)


def _expected_values(X: pd.DataFrame) -> pd.Series:
    """Training medians for continuous predictors, training mode for is_male."""
    values = X[CONTINUOUS].median(skipna=True)
    for col in BINARY_PREDICTORS:
        counts = X[col].value_counts(dropna=True)
        # Most frequent; ties resolve to the smallest value, as in SimpleImputer.
        values[col] = counts[counts == counts.max()].index.min()
    return values.loc[config.predictors()]


def test_split_is_patient_level_and_complete(split, prepared):
    _, _, ids = prepared
    train, test = set(split.ids_train), set(split.ids_test)
    assert not train & test
    assert train | test == set(ids)
    assert len(test) == round(len(ids) * EVAL["test_size"])


@pytest.mark.parametrize("model", list(MODELS))
def test_imputer_uses_training_rows_only(split, model):
    fitted = MODELS[model](0).fit(split.X_train, split.y_train)
    values = imputation_values(fitted)
    pd.testing.assert_series_equal(
        values, _expected_values(split.X_train), check_names=False
    )
    # The pooled medians differ, so this test would catch fitting on all rows.
    pooled = pd.concat([split.X_train, split.X_test])[CONTINUOUS].median()
    assert not np.allclose(values[CONTINUOUS], pooled)


@pytest.mark.parametrize("model", list(MODELS))
def test_same_preprocessing_and_18_features_for_both_models(split, model):
    fitted = MODELS[model](0).fit(split.X_train, split.y_train)
    out = fitted[:-1].transform(split.X_test)
    assert list(out.columns) == config.predictors()
    assert not out.isna().any().any()
    other = "lightgbm" if model == "logistic_regression" else "logistic_regression"
    other_fit = MODELS[other](0).fit(split.X_train, split.y_train)
    pd.testing.assert_series_equal(
        imputation_values(fitted), imputation_values(other_fit)
    )


def test_is_male_imputed_to_observed_category_not_half():
    """With a 50/50 split of observed sex, a median imputer would return 0.5.
    The most-frequent imputer must return an observed category (0 or 1)."""
    predictors = config.predictors()
    n = 20
    X = pd.DataFrame(
        {c: np.linspace(1.0, 2.0, n) for c in predictors if c != "is_male"}
    )
    X["is_male"] = [0.0, 1.0] * (n // 2)
    X.loc[[0, 1], "is_male"] = np.nan  # leaves 9 zeros and 9 ones
    X = X[predictors]
    y = pd.Series([0, 1] * (n // 2))
    assert X["is_male"].median() == 0.5  # what the old imputer would have used
    for model in MODELS:
        fitted = MODELS[model](0).fit(X, y)
        assert imputation_values(fitted)["is_male"] in (0.0, 1.0)
        imputed = fitted.named_steps["impute"].transform(fitted[0].transform(X))
        assert set(imputed["is_male"].unique()) <= {0.0, 1.0}


@pytest.mark.parametrize("model", list(MODELS))
def test_preprocessing_ignores_labels(split, model):
    shuffled = split.y_train.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert not shuffled.equals(split.y_train)
    a = MODELS[model](0).fit(split.X_train, split.y_train)
    b = MODELS[model](0).fit(split.X_train, shuffled)
    pd.testing.assert_series_equal(imputation_values(a), imputation_values(b))
    if "scale" in a.named_steps:
        np.testing.assert_array_equal(
            a.named_steps["scale"].mean_, b.named_steps["scale"].mean_
        )


def test_imputer_is_refitted_inside_each_cv_fold(split):
    cv = StratifiedKFold(n_splits=EVAL["cv_folds"], shuffle=True, random_state=0)
    result = cross_validate(
        MODELS["logistic_regression"](0),
        split.X_train,
        split.y_train,
        cv=cv,
        scoring="roc_auc",
        return_estimator=True,
        return_indices=True,
    )
    for est, train_idx in zip(result["estimator"], result["indices"]["train"]):
        expected = _expected_values(split.X_train.iloc[train_idx])
        pd.testing.assert_series_equal(
            imputation_values(est), expected, check_names=False
        )
