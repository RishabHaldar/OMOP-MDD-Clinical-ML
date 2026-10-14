"""The synthetic generator: determinism, synthetic vocabulary, null process."""

from __future__ import annotations

import pandas as pd
import pytest

from omop_mdd import config
from omop_mdd.synthetic import SyntheticConfig, generate

RECORD_TABLES = ["person", "observation_period", "visit_occurrence", "measurement"]


def test_same_seed_gives_identical_tables():
    a = generate(SyntheticConfig(seed=7, n_persons=200))
    b = generate(SyntheticConfig(seed=7, n_persons=200))
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_different_seed_gives_different_measurements():
    a = generate(SyntheticConfig(seed=7, n_persons=200))
    b = generate(SyntheticConfig(seed=8, n_persons=200))
    assert not a["measurement"].equals(b["measurement"])


def test_null_process_records_do_not_depend_on_case_status():
    """With signal = 0, changing who is a case leaves every non-condition
    table byte-identical: the generator cannot encode the label in them."""
    a = generate(SyntheticConfig(seed=11, n_persons=300, signal=0.0, label_seed=1))
    b = generate(SyntheticConfig(seed=11, n_persons=300, signal=0.0, label_seed=2))
    assert not a["_truth"]["is_case"].equals(b["_truth"]["is_case"])
    for name in RECORD_TABLES:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_signal_changes_measurements_only_through_case_status():
    """Positive control for the test above: with a signal, different case
    assignments do change measurement values."""
    a = generate(SyntheticConfig(seed=11, n_persons=300, signal=1.0, label_seed=1))
    b = generate(SyntheticConfig(seed=11, n_persons=300, signal=1.0, label_seed=2))
    pd.testing.assert_frame_equal(a["person"], b["person"])
    assert not a["measurement"]["value_as_number"].equals(
        b["measurement"]["value_as_number"]
    )


def test_clinical_concepts_are_synthetic(tables):
    assert (tables["condition_occurrence"]["condition_concept_id"] >= 2e9).all()
    assert (tables["measurement"]["measurement_concept_id"] >= 2e9).all()
    units = set(tables["measurement"]["unit_concept_id"])
    allowed_units = set(config.concept_sets()["units"].values()) | {0}
    assert units <= allowed_units
    assert set(tables["person"]["gender_concept_id"]) <= {8507, 8532, 0}


def test_omop_column_names(tables):
    expected = {
        "person": {"person_id", "gender_concept_id", "year_of_birth"},
        "measurement": {
            "person_id",
            "measurement_concept_id",
            "measurement_date",
            "value_as_number",
            "unit_concept_id",
        },
        "condition_occurrence": {
            "person_id",
            "condition_concept_id",
            "condition_start_date",
        },
    }
    for table, cols in expected.items():
        assert cols <= set(tables[table].columns)


@pytest.mark.parametrize("unit_kind", ["alternate", "unmapped"])
def test_generator_emits_units_the_sql_must_handle(tables, unit_kind):
    cs = config.concept_sets()
    units = tables["measurement"]["unit_concept_id"]
    if unit_kind == "unmapped":
        assert (units == 0).any()
    else:
        alt_ids = {
            cs["units"][spec["alternate_unit"]["unit"]]
            for spec in cs["labs"].values()
            if spec["alternate_unit"]
        }
        assert units.isin(alt_ids).any()
