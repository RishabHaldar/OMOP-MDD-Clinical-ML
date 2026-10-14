"""Repeated-simulation sanity check under the null (signal = 0).

    python -m omop_mdd.null_check --reps 30

Not part of pytest: a single small-sample AUROC is too variable to assert on.
Across many independent synthetic datasets with no generated signal, the
leakage-safe pipeline's AUROC should be centred near 0.5. A clear, consistent
shift away from 0.5 would point to label information entering through the
pipeline or through cohort construction.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from omop_mdd.run import run


def simulate(reps: int, first_seed: int) -> pd.DataFrame:
    rows = []
    for seed in range(first_seed, first_seed + reps):
        res = run(
            seed=seed, signal=0.0, out=None, make_figures=False, with_importance=False
        )
        for name in res["cv"]:
            rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "cv_auroc": res["cv"][name]["auroc_mean"],
                    "test_auroc": res["test"][name]["auroc"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Null-signal repeated simulation")
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=10_000)
    args = parser.parse_args()
    df = simulate(args.reps, args.first_seed)
    summary = df.groupby("model")[["cv_auroc", "test_auroc"]].agg(
        [
            "mean",
            "std",
            lambda s: np.percentile(s, 2.5),
            lambda s: np.percentile(s, 97.5),
        ]
    )
    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in zip(
            summary.columns.get_level_values(0),
            ["mean", "sd", "p2.5", "p97.5"] * 2,
        )
    ]
    print(f"Null simulations: {args.reps} synthetic datasets, signal = 0")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
