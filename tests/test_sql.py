"""The reconstructed SQL, executed against synthetic OMOP tables in DuckDB."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omop_mdd import config
from omop_mdd.extract import build_database, run_sql
from omop_mdd.synthetic import DEPRESSION_DESCENDANTS, _vocabulary

N = config.analysis()["cohort"]["n_per_cohort"]


def _age_at(index_dates: pd.Series, person: pd.DataFrame) -> pd.Series:
    birth = pd.to_datetime(
        dict(
            year=person["year_of_birth"],
            month=person["month_of_birth"],
            day=person["day_of_birth"],
        )
    )
    return (pd.to_datetime(index_dates) - birth).dt.days / 365.25


# --- Full synthetic population ------------------------------------------------


def test_cohort_sizes_and_one_row_per_person(dataset):
    assert dataset["cohort"].value_counts().to_dict() == {"case": N, "comparison": N}
    assert dataset["person_id"].is_unique


def test_cohorts_do_not_overlap(con):
    overlap = con.execute(
        "SELECT COUNT(*) FROM case_cohort c JOIN comparison_cohort k USING (person_id)"
    ).fetchone()[0]
    assert overlap == 0


def test_same_age_rule_and_bounds_in_both_cohorts(con, tables):
    cfg = config.analysis()["cohort"]
    for table in ["case_cohort", "comparison_cohort"]:
        cohort = con.execute(
            f"SELECT person_id, index_date, age_at_index FROM {table}"
        ).df()
        assert cohort["age_at_index"].between(cfg["min_age"], cfg["max_age"]).all()
        person = (
            tables["person"]
            .set_index("person_id")
            .loc[cohort["person_id"]]
            .reset_index()
        )
        expected = _age_at(cohort["index_date"], person)
        np.testing.assert_allclose(cohort["age_at_index"], expected, rtol=0, atol=1e-9)


def test_case_index_is_first_qualifying_diagnosis(con, tables):
    cases = con.execute("SELECT person_id, index_date FROM case_cohort").df()
    co = tables["condition_occurrence"]
    dx = co[co["condition_concept_id"].isin([2000000100, *DEPRESSION_DESCENDANTS])]
    first = dx.groupby("person_id")["condition_start_date"].min()
    expected = first.loc[cases["person_id"]].dt.date.to_numpy()
    assert (pd.to_datetime(cases["index_date"]).dt.date.to_numpy() == expected).all()


def test_descendant_only_cases_are_included(con, tables):
    co = tables["condition_occurrence"]
    root_holders = set(co.loc[co["condition_concept_id"] == 2000000100, "person_id"])
    descendant_only = (
        set(
            co.loc[co["condition_concept_id"].isin(DEPRESSION_DESCENDANTS), "person_id"]
        )
        - root_holders
    )
    cases = set(con.execute("SELECT person_id FROM case_cohort").df()["person_id"])
    eligible = descendant_only & cases
    assert eligible, "expected some descendant-only cases in the synthetic data"


def test_comparison_index_is_first_visit_and_no_exclusions(con, tables):
    comp = con.execute("SELECT person_id, index_date FROM comparison_cohort").df()
    first_visit = (
        tables["visit_occurrence"].groupby("person_id")["visit_start_date"].min()
    )
    expected = first_visit.loc[comp["person_id"]].dt.date.to_numpy()
    assert (pd.to_datetime(comp["index_date"]).dt.date.to_numpy() == expected).all()
    co = tables["condition_occurrence"]
    excluded = set(
        co.loc[
            co["condition_concept_id"].isin(
                [2000000100, 2000000110, *DEPRESSION_DESCENDANTS]
            ),
            "person_id",
        ]
    )
    assert not excluded & set(comp["person_id"])


def test_selection_uses_total_measurement_count(con):
    features = con.execute("SELECT * FROM patient_features").df()
    selected = set(
        con.execute("SELECT person_id FROM analysis_dataset").df()["person_id"]
    )
    for cohort, g in features.groupby("cohort"):
        g = g.assign(selected=g["person_id"].isin(selected))
        assert g["selected"].sum() == N
        # Every selected person has at least as many measurements as every
        # unselected person in the same cohort.
        if (~g["selected"]).any():
            assert (
                g.loc[g["selected"], "n_measurements"].min()
                >= g.loc[~g["selected"], "n_measurements"].max()
            )


def test_only_listed_units_reach_lab_measurements(con):
    raw = con.execute("""
        SELECT COUNT(*) FROM lab_measurements l
        """).fetchone()[0]
    eligible = con.execute("""
        SELECT COUNT(*)
        FROM measurement m
        JOIN (SELECT person_id FROM case_cohort UNION ALL
              SELECT person_id FROM comparison_cohort) c USING (person_id)
        JOIN lab_concept lc ON lc.concept_id = m.measurement_concept_id
        JOIN unit_conversion uc
            ON uc.analyte = lc.analyte AND uc.unit_concept_id = m.unit_concept_id
        """).fetchone()[0]
    unmapped = con.execute("""
        SELECT COUNT(*)
        FROM measurement m
        JOIN (SELECT person_id FROM case_cohort UNION ALL
              SELECT person_id FROM comparison_cohort) c USING (person_id)
        WHERE m.unit_concept_id = 0
        """).fetchone()[0]
    assert unmapped > 0
    assert raw == eligible


# --- Hand-built minimal database ---------------------------------------------


def _minimal_tables(gender_case: int = 0) -> dict[str, pd.DataFrame]:
    """Two persons: one case (person 1), one comparison (person 2), both aged
    40 at index, with measurements in canonical, alternate and unmapped units."""
    cs = config.concept_sets()
    units = cs["units"]
    labs = cs["labs"]
    d0 = pd.Timestamp("2020-01-01")
    d1 = pd.Timestamp("2020-06-01")
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "gender_concept_id": [gender_case, 8507],
            "year_of_birth": [1980, 1980],
            "month_of_birth": [1, 1],
            "day_of_birth": [1, 1],
        }
    )
    observation_period = pd.DataFrame(
        {
            "observation_period_id": [1, 2],
            "person_id": [1, 2],
            "observation_period_start_date": [pd.Timestamp("2019-01-01")] * 2,
            "observation_period_end_date": [pd.Timestamp("2021-12-31")] * 2,
        }
    )
    visit_occurrence = pd.DataFrame(
        {
            "visit_occurrence_id": [1, 2, 3, 4],
            "person_id": [1, 1, 2, 2],
            "visit_start_date": [d0, d1, d0, d1],
            "visit_end_date": [d0, d1, d0, d1],
        }
    )
    condition_occurrence = pd.DataFrame(
        {
            "condition_occurrence_id": [1],
            "person_id": [1],
            "condition_concept_id": [DEPRESSION_DESCENDANTS[0]],
            "condition_start_date": [d0],
        }
    )
    g = labs["glucose"]["concept_id"]
    a1c = labs["hba1c"]["concept_id"]
    measurement = pd.DataFrame(
        [
            (1, 1, g, d0, 100.0, units["mg/dL"]),
            (2, 1, g, d1, 100.0 / 18.016, units["mmol/L"]),
            (3, 1, g, d1, 9999.0, 0),  # unmapped unit: must be excluded
            (4, 1, a1c, d0, 48.0, units["mmol/mol"]),
            (5, 2, g, d0, 90.0, units["mg/dL"]),
        ],
        columns=[
            "measurement_id",
            "person_id",
            "measurement_concept_id",
            "measurement_date",
            "value_as_number",
            "unit_concept_id",
        ],
    )
    concept, concept_ancestor = _vocabulary()
    return {
        "person": person,
        "observation_period": observation_period,
        "visit_occurrence": visit_occurrence,
        "condition_occurrence": condition_occurrence,
        "measurement": measurement,
        "concept": concept,
        "concept_ancestor": concept_ancestor,
    }


def _run_minimal(tables, **overrides) -> pd.DataFrame:
    params = dict(config.analysis()["cohort"])
    params.update(overrides)
    con = build_database(tables, params)
    out = run_sql(con).set_index("person_id")
    con.close()
    return out


def test_unit_conversion_and_exclusion():
    out = _run_minimal(_minimal_tables())
    # 100 mg/dL and 100/18.016 mmol/L both become 100 mg/dL; 9999 (unit 0) dropped.
    assert out.loc[1, "glucose_median"] == pytest.approx(100.0)
    # 48 mmol/mol IFCC -> 0.09148 * 48 + 2.152 = 6.543 % NGSP
    assert out.loc[1, "hba1c_median"] == pytest.approx(0.09148 * 48 + 2.152)
    assert out.loc[1, "n_measurements"] == 3
    assert out.loc[1, "n_distinct_analytes"] == 2


def test_unknown_gender_is_missing_not_female():
    out = _run_minimal(_minimal_tables(gender_case=0))
    assert pd.isna(out.loc[1, "is_male"])
    assert out.loc[2, "is_male"] == 1


def test_lab_window_relative_to_index():
    out = _run_minimal(
        _minimal_tables(), window_days_before_index=0, window_days_after_index=30
    )
    # Only the index-date glucose (100 mg/dL) and HbA1c remain for person 1.
    assert out.loc[1, "n_measurements"] == 2


def test_both_cohorts_use_age_at_index():
    out = _run_minimal(_minimal_tables())
    assert out.loc[1, "cohort"] == "case"
    assert out.loc[2, "cohort"] == "comparison"
    # Both born 1980-01-01, both indexed 2020-01-01.
    assert out.loc[1, "age"] == pytest.approx(out.loc[2, "age"])
    assert out.loc[1, "age"] == pytest.approx(14610 / 365.25)
