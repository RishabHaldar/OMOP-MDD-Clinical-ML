"""Label assignment and cohort integrity checks."""

from __future__ import annotations

import pandas as pd

from omop_mdd.features import select_predictors

COHORT_LABELS = {"case": 1, "comparison": 0}


class CohortIntegrityError(ValueError):
    """Raised when the extracted cohorts violate an integrity check."""


def validate_cohorts(df: pd.DataFrame, n_per_cohort: int | None = None) -> None:
    """Check identifiers and cohort membership before any modeling."""
    if df["person_id"].isna().any():
        raise CohortIntegrityError("Missing person_id values.")
    duplicated = df["person_id"].duplicated()
    if duplicated.any():
        # A person in both cohorts also appears here.
        raise CohortIntegrityError(
            f"{int(duplicated.sum())} duplicated person_id value(s)."
        )
    unknown = set(df["cohort"].unique()) - set(COHORT_LABELS)
    if unknown:
        raise CohortIntegrityError(f"Unknown cohort value(s): {sorted(unknown)}")
    counts = df["cohort"].value_counts()
    for cohort in COHORT_LABELS:
        if counts.get(cohort, 0) == 0:
            raise CohortIntegrityError(f"No patients in cohort '{cohort}'.")
        if n_per_cohort is not None and counts[cohort] != n_per_cohort:
            raise CohortIntegrityError(
                f"Cohort '{cohort}' has {counts[cohort]} patients, "
                f"expected {n_per_cohort}."
            )


def prepare(
    df: pd.DataFrame, n_per_cohort: int | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, person_id). Labels come only from the SQL ``cohort`` column."""
    validate_cohorts(df, n_per_cohort)
    df = df.sort_values("person_id").reset_index(drop=True)
    X = select_predictors(df)
    y = df["cohort"].map(COHORT_LABELS).astype(int).rename("label")
    return X, y, df["person_id"]
