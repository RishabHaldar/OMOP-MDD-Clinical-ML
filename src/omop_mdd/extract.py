"""Run the reconstructed SQL in sql/ against synthetic OMOP tables in DuckDB."""

from __future__ import annotations

import duckdb
import pandas as pd

from omop_mdd import config
from omop_mdd.synthetic import SyntheticConfig, generate

SQL_FILES = [
    "01_case_cohort.sql",
    "02_comparison_cohort.sql",
    "03_lab_measurements.sql",
    "04_patient_features.sql",
    "05_completeness_selection.sql",
]

OMOP_TABLES = [
    "person",
    "observation_period",
    "visit_occurrence",
    "condition_occurrence",
    "measurement",
    "concept",
    "concept_ancestor",
]


def _support_tables(params: dict) -> dict[str, pd.DataFrame]:
    """Concept-set, unit-conversion and parameter tables used by the SQL."""
    cs = config.concept_sets()
    members = [
        (name, cid) for name, spec in cs["conditions"].items() for cid in spec["roots"]
    ]
    concept_set_member = pd.DataFrame(members, columns=["concept_set", "concept_id"])

    lab_concept = pd.DataFrame(
        [(name, spec["concept_id"]) for name, spec in cs["labs"].items()],
        columns=["analyte", "concept_id"],
    )

    conversions = []
    for name, spec in cs["labs"].items():
        conversions.append((name, cs["units"][spec["canonical_unit"]], 1.0, 0.0))
        alt = spec["alternate_unit"]
        if alt is not None:
            conversions.append(
                (
                    name,
                    cs["units"][alt["unit"]],
                    float(alt["multiplier"]),
                    float(alt["offset"]),
                )
            )
    unit_conversion = pd.DataFrame(
        conversions, columns=["analyte", "unit_concept_id", "multiplier", "offset"]
    )

    analysis_params = pd.DataFrame(
        [
            {
                "min_age": float(params["min_age"]),
                "max_age": float(params["max_age"]),
                "n_per_cohort": int(params["n_per_cohort"]),
                "window_days_before_index": params["window_days_before_index"],
                "window_days_after_index": params["window_days_after_index"],
            }
        ]
    ).astype(
        {
            "window_days_before_index": "Int64",
            "window_days_after_index": "Int64",
        }
    )
    return {
        "concept_set_member": concept_set_member,
        "lab_concept": lab_concept,
        "unit_conversion": unit_conversion,
        "analysis_params": analysis_params,
    }


def build_database(
    tables: dict[str, pd.DataFrame], params: dict | None = None
) -> duckdb.DuckDBPyConnection:
    """Load OMOP + support tables into an in-memory DuckDB database.

    Only the OMOP tables are loaded; the generator's ``_truth`` table is never
    visible to the SQL.
    """
    params = params or config.analysis()["cohort"]
    con = duckdb.connect(database=":memory:")
    for name in OMOP_TABLES:
        df = tables[name]
        con.register("_tmp", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _tmp")
        con.unregister("_tmp")
    for name, df in _support_tables(params).items():
        con.register("_tmp", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _tmp")
        con.unregister("_tmp")
    # Dates arrive from pandas as TIMESTAMP; cast to DATE as in the OMOP CDM.
    for table, cols in {
        "observation_period": [
            "observation_period_start_date",
            "observation_period_end_date",
        ],
        "visit_occurrence": ["visit_start_date", "visit_end_date"],
        "condition_occurrence": ["condition_start_date"],
        "measurement": ["measurement_date"],
    }.items():
        for col in cols:
            con.execute(f"ALTER TABLE {table} ALTER {col} TYPE DATE")
    return con


def run_sql(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Execute sql/01-05 in order and return the analysis dataset."""
    for filename in SQL_FILES:
        con.execute((config.SQL_DIR / filename).read_text(encoding="utf-8"))
    return con.execute("SELECT * FROM analysis_dataset").df()


def extract(
    synthetic_cfg: SyntheticConfig | None = None, params: dict | None = None
) -> tuple[pd.DataFrame, duckdb.DuckDBPyConnection]:
    """Generate synthetic OMOP data, run the SQL, return (dataset, connection)."""
    tables = generate(synthetic_cfg)
    con = build_database(tables, params)
    return run_sql(con), con
