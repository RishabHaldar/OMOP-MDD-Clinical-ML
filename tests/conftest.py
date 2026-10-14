"""Shared fixtures. All data is synthetic and generated at test time."""

from __future__ import annotations

import pandas as pd
import pytest

from omop_mdd import config
from omop_mdd.data import prepare
from omop_mdd.extract import build_database, run_sql
from omop_mdd.synthetic import SyntheticConfig, generate

SEED = 2026


@pytest.fixture(scope="session")
def tables() -> dict[str, pd.DataFrame]:
    return generate(SyntheticConfig(seed=SEED, signal=0.5))


@pytest.fixture(scope="session")
def con(tables):
    connection = build_database(tables)
    run_sql(connection)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def dataset(con) -> pd.DataFrame:
    return con.execute("SELECT * FROM analysis_dataset").df()


@pytest.fixture(scope="session")
def prepared(dataset):
    n = config.analysis()["cohort"]["n_per_cohort"]
    return prepare(dataset, n_per_cohort=n)
