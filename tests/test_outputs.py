"""End-to-end run: output schema and files."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from omop_mdd import config
from omop_mdd.models import MODELS
from omop_mdd.run import run


@pytest.fixture(scope="module")
def outdir(tmp_path_factory):
    out = tmp_path_factory.mktemp("outputs")
    run(seed=2026, out=out)
    return out


def test_expected_files(outdir):
    for name in [
        "metrics.json",
        "test_predictions.csv",
        "importance_logistic_regression.csv",
        "importance_lightgbm.csv",
        "roc_test.png",
        "confusion_test.png",
        "permutation_importance.png",
        "lr_coefficients.png",
    ]:
        assert (outdir / name).stat().st_size > 0, name


def test_metrics_schema(outdir):
    m = json.loads((outdir / "metrics.json").read_text())
    assert m["data"] == "SYNTHETIC"
    assert set(m["cv"]) == set(m["test"]) == set(MODELS)
    for name in MODELS:
        t = m["test"][name]
        assert 0.0 <= t["auroc"] <= 1.0
        lo, hi = t["auroc_95ci_bootstrap"]
        assert lo <= t["auroc"] <= hi
        assert set(t["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}
        assert sum(t["confusion_matrix"].values()) == t["n_test"]
        for key in ["precision", "recall", "f1"]:
            assert 0.0 <= t[key] <= 1.0
        cv = m["cv"][name]
        assert len(cv["auroc_folds"]) == config.analysis()["evaluation"]["cv_folds"]


def test_predictions_schema(outdir):
    p = pd.read_csv(outdir / "test_predictions.csv")
    assert list(p.columns) == ["person_id", "label"] + [f"prob_{n}" for n in MODELS]
    assert p["person_id"].is_unique


def test_importance_covers_all_predictors(outdir):
    for name in MODELS:
        imp = pd.read_csv(outdir / f"importance_{name}.csv")
        assert list(imp["feature"]) == config.predictors()
