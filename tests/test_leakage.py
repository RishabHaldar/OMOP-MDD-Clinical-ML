"""Only the 18 allowlisted predictors can reach a model."""

from __future__ import annotations

import pytest

from omop_mdd import config
from omop_mdd.features import LeakageError, check_allowlist, select_predictors
from omop_mdd.models import MODELS


def test_allowlist_has_18_predictors_and_no_identifiers():
    predictors = config.predictors()
    assert len(predictors) == 18 == len(set(predictors))
    assert not set(predictors) & set(config.features()["non_predictors"])


def test_feature_matrix_is_exactly_the_allowlist(prepared):
    X, _, _ = prepared
    assert list(X.columns) == config.predictors()


def test_select_predictors_drops_identifiers(dataset):
    X = select_predictors(dataset)
    for col in ["person_id", "cohort", "n_measurements", "n_distinct_analytes"]:
        assert col not in X.columns


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("extra", ["group", "label", "person_id", "n_measurements"])
def test_pipeline_rejects_forbidden_columns(prepared, model, extra):
    X, y, ids = prepared
    X_bad = X.copy()
    X_bad[extra] = y.to_numpy() if extra != "person_id" else ids.to_numpy()
    with pytest.raises(LeakageError):
        MODELS[model](0).fit(X_bad, y)


@pytest.mark.parametrize("model", list(MODELS))
def test_pipeline_rejects_unknown_extra_column(prepared, model):
    X, y, _ = prepared
    X_bad = X.assign(target_encoded=y.to_numpy())
    with pytest.raises(LeakageError):
        MODELS[model](0).fit(X_bad, y)


def test_reordered_or_missing_columns_are_rejected():
    cols = config.predictors()
    with pytest.raises(LeakageError):
        check_allowlist(list(reversed(cols)))
    with pytest.raises(LeakageError):
        check_allowlist(cols[:-1])


@pytest.mark.parametrize("model", list(MODELS))
def test_prediction_also_checks_columns(prepared, model):
    X, y, _ = prepared
    fitted = MODELS[model](0).fit(X, y)
    with pytest.raises(LeakageError):
        fitted.predict_proba(X.assign(group=y.to_numpy()))
