"""Preprocessing is fitted on training data only, without labels."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_validate

from omop_mdd import config
from omop_mdd.evaluate import split_patients
from omop_mdd.models import MODELS

EVAL = config.analysis()["evaluation"]


@pytest.fixture(scope="module")
def split(prepared):
    X, y, ids = prepared
    return split_patients(X, y, ids, EVAL["test_size"], seed=0)


def test_split_is_patient_level_and_complete(split, prepared):
    _, _, ids = prepared
    train, test = set(split.ids_train), set(split.ids_test)
    assert not train & test
    assert train | test == set(ids)
    assert len(test) == round(len(ids) * EVAL["test_size"])


@pytest.mark.parametrize("model", list(MODELS))
def test_imputer_uses_training_medians_only(split, model):
    fitted = MODELS[model](0).fit(split.X_train, split.y_train)
    stats = fitted.named_steps["impute"].statistics_
    expected = np.nanmedian(split.X_train.to_numpy(), axis=0)
    np.testing.assert_allclose(stats, expected)
    # The held-out medians are different, so this test would catch fitting on
    # all rows.
    pooled = np.nanmedian(
        np.vstack([split.X_train.to_numpy(), split.X_test.to_numpy()]), axis=0
    )
    assert not np.allclose(stats, pooled)


@pytest.mark.parametrize("model", list(MODELS))
def test_preprocessing_ignores_labels(split, model):
    shuffled = split.y_train.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert not shuffled.equals(split.y_train)
    a = MODELS[model](0).fit(split.X_train, split.y_train)
    b = MODELS[model](0).fit(split.X_train, shuffled)
    np.testing.assert_array_equal(
        a.named_steps["impute"].statistics_, b.named_steps["impute"].statistics_
    )
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
    X = split.X_train.to_numpy()
    for est, train_idx in zip(result["estimator"], result["indices"]["train"]):
        expected = np.nanmedian(X[train_idx], axis=0)
        np.testing.assert_allclose(est.named_steps["impute"].statistics_, expected)
