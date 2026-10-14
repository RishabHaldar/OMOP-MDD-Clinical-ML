"""End-to-end synthetic demonstration.

    python -m omop_mdd.run --seed 2026 --signal 0.5 --out outputs

Steps: generate synthetic OMOP tables -> run sql/01-05 in DuckDB -> validate
cohorts -> patient-level split -> training-only CV -> fit -> single held-out
evaluation -> figures and aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from omop_mdd import config, plots
from omop_mdd.data import prepare
from omop_mdd.evaluate import (
    cross_validate_training,
    evaluate_test,
    importance,
    split_patients,
)
from omop_mdd.extract import extract
from omop_mdd.models import MODELS
from omop_mdd.synthetic import SyntheticConfig

PACKAGES = ["numpy", "pandas", "scikit-learn", "lightgbm", "duckdb", "matplotlib"]


def cohort_summary(df: pd.DataFrame) -> dict:
    """Aggregate, descriptive summary of the extracted synthetic cohorts."""
    predictors = config.predictors()
    summary = {}
    for cohort, g in df.groupby("cohort"):
        summary[cohort] = {
            "n": int(len(g)),
            "age_mean": float(g["age"].mean()),
            "n_measurements_median": float(g["n_measurements"].median()),
            "n_distinct_analytes_median": float(g["n_distinct_analytes"].median()),
            "missing_fraction": {c: float(g[c].isna().mean()) for c in predictors},
        }
    return summary


def run(
    seed: int = 2026,
    signal: float = 0.5,
    out: Path | None = Path("outputs"),
    make_figures: bool = True,
    with_importance: bool = True,
) -> dict:
    eval_cfg = config.analysis()["evaluation"]
    cohort_cfg = config.analysis()["cohort"]

    dataset, con = extract(SyntheticConfig(seed=seed, signal=signal), cohort_cfg)
    con.close()
    X, y, ids = prepare(dataset, n_per_cohort=cohort_cfg["n_per_cohort"])
    split = split_patients(X, y, ids, eval_cfg["test_size"], seed)

    results: dict = {
        "data": "SYNTHETIC",
        "seed": seed,
        "signal": signal,
        "split": {
            "method": "stratified patient-level train/test split",
            "test_size": eval_cfg["test_size"],
            "n_train": int(len(split.y_train)),
            "n_test": int(len(split.y_test)),
            "n_test_cases": int(split.y_test.sum()),
        },
        "cv": {},
        "test": {},
        "cohorts": cohort_summary(dataset),
        "environment": {
            "python": platform.python_version(),
            **{p: version(p) for p in PACKAGES},
        },
    }

    probs, importances, predictions = {}, {}, {"person_id": split.ids_test}
    predictions["label"] = split.y_test
    for name, factory in MODELS.items():
        pipe = factory(seed)
        results["cv"][name] = cross_validate_training(
            pipe, split.X_train, split.y_train, eval_cfg["cv_folds"], seed
        )
        fitted = pipe.fit(split.X_train, split.y_train)
        metrics, prob = evaluate_test(
            fitted,
            split.X_test,
            split.y_test,
            eval_cfg["decision_threshold"],
            eval_cfg["bootstrap_resamples"],
            seed,
        )
        results["test"][name] = metrics
        probs[name] = prob
        predictions[f"prob_{name}"] = prob
        if with_importance:
            importances[name] = importance(fitted, split.X_test, split.y_test, seed)

    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
        pd.DataFrame(predictions).to_csv(out / "test_predictions.csv", index=False)
        for name, imp in importances.items():
            imp.to_csv(out / f"importance_{name}.csv", index=False)
        if make_figures and with_importance:
            plots.roc_figure(split.y_test, probs, out / "roc_test.png")
            plots.confusion_figure(results["test"], out / "confusion_test.png")
            plots.permutation_figure(importances, out / "permutation_importance.png")
            plots.coefficient_figure(
                importances["logistic_regression"], out / "lr_coefficients.png"
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--signal",
        type=float,
        default=0.5,
        help="Synthetic case/control shift in SD units (0 = no signal).",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    results = run(args.seed, args.signal, args.out, not args.no_figures)
    for name in MODELS:
        cv = results["cv"][name]
        test = results["test"][name]
        lo, hi = test["auroc_95ci_bootstrap"]
        print(
            f"{name:20s} CV AUROC {cv['auroc_mean']:.3f} (SD {cv['auroc_sd']:.3f}) | "
            f"test AUROC {test['auroc']:.3f} [{lo:.3f}, {hi:.3f}]"
        )
    print(f"Outputs written to {args.out}/ (SYNTHETIC data).")


if __name__ == "__main__":
    main()
