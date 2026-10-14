"""Cohort-label integrity."""

from __future__ import annotations

import pandas as pd
import pytest

from omop_mdd import config
from omop_mdd.data import CohortIntegrityError, prepare, validate_cohorts

N = config.analysis()["cohort"]["n_per_cohort"]


def test_label_counts(prepared):
    _, y, _ = prepared
    assert y.value_counts().to_dict() == {1: N, 0: N}


def test_labels_follow_cohort_column(dataset):
    X, y, ids = prepare(dataset, n_per_cohort=N)
    cohort = dataset.set_index("person_id").loc[ids, "cohort"].to_numpy()
    assert ((cohort == "case") == (y.to_numpy() == 1)).all()


def test_sql_labels_agree_with_generator_truth(prepared, tables):
    _, y, ids = prepared
    truth = tables["_truth"].set_index("person_id").loc[ids, "is_case"]
    assert (truth.to_numpy() == (y.to_numpy() == 1)).all()


def test_no_duplicate_or_overlapping_ids(prepared):
    _, _, ids = prepared
    assert ids.is_unique


def test_duplicate_person_is_rejected(dataset):
    dup = pd.concat([dataset, dataset.iloc[[0]]], ignore_index=True)
    with pytest.raises(CohortIntegrityError):
        validate_cohorts(dup)


def test_person_in_both_cohorts_is_rejected(dataset):
    row = dataset[dataset["cohort"] == "case"].iloc[[0]].copy()
    row["cohort"] = "comparison"
    with pytest.raises(CohortIntegrityError):
        validate_cohorts(pd.concat([dataset, row], ignore_index=True))


def test_unknown_cohort_value_is_rejected(dataset):
    bad = dataset.copy()
    bad.loc[0, "cohort"] = "control"
    with pytest.raises(CohortIntegrityError):
        validate_cohorts(bad)


def test_wrong_cohort_size_is_rejected(dataset):
    with pytest.raises(CohortIntegrityError):
        validate_cohorts(dataset.iloc[1:], n_per_cohort=N)
