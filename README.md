# OMOP-MDD-Clinical-ML

<!--
REVIEW NOTE (remove before publishing):
Research-facing sections (background, research question, historical study,
role, methods narrative, results, limitations, citation) are to be supplied
separately. Only the technical sections below were drafted here.
-->

> **All data in this repository is synthetic.** The repository contains no patient data, no institutional SQL, and no results from the historical Washington University study.

**Scope of the demonstration:**

- **Retrospective classification, not prediction.** By default, laboratory medians use every measurement in the observation period, including measurements taken after the index date. This is retrospective case-control classification, not prospective prediction of depression onset.
- **Minimal tables.** The generated database has minimal OMOP-shaped tables with only selected columns. It is not a complete or validated OMOP CDM installation.
- **Programmed signal.** The synthetic case/control difference is deliberately programmed into glucose, HbA1c, BMI, HDL and leukocyte values. Feature-importance results reflect that design and do not establish biological associations.

## Repository layout

```
OMOP-MDD-Clinical-ML/
├── config/
│   ├── analysis.yaml         # cohort parameters (age bounds, n per cohort, window) and evaluation settings
│   ├── concept_sets.yaml     # SYNTHETIC concept IDs, units and conversions used by the SQL
│   └── features.yaml         # the 18-predictor allowlist
├── sql/                      # newly reconstructed OMOP SQL (not the original queries) — see sql/README.md
├── src/omop_mdd/
│   ├── synthetic.py          # synthetic OMOP tables
│   ├── extract.py            # runs sql/ in DuckDB
│   ├── data.py               # labels and cohort integrity checks
│   ├── features.py           # allowlist enforcement
│   ├── models.py             # logistic regression and LightGBM pipelines
│   ├── evaluate.py           # split, training-only CV, held-out evaluation
│   ├── plots.py              # figures
│   ├── run.py                # end-to-end command
│   └── null_check.py         # optional repeated-simulation check
├── tests/
├── docs/reproducibility.md
├── .github/workflows/ci.yml  # black --check + pytest
├── pyproject.toml
└── requirements.txt
```

## Quick start

Requires Python 3.13. From the repository directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

python -m omop_mdd.run --seed 2026     # writes outputs/
pytest                                 # test suite
black --check .                        # formatting
```

`docs/reproducibility.md` covers the full pipeline, outputs, tests and known limitations.

## Evaluation design (implemented)

- Stratified **75/25** patient-level train/test split with a fixed seed. The historical study used 80/20.
- Predictors restricted to an explicit 18-column allowlist: age, sex and 16 laboratory medians. Identifiers, cohort labels and completeness counts are rejected.
- Identical imputation for both models, fitted on training rows only inside a scikit-learn Pipeline: most frequent value for `is_male`, median for the other 17 predictors. Logistic regression also standardizes.
- 5-fold stratified cross-validation on the training set only.
- One held-out evaluation per model:
  - AUROC with a bootstrap 95% CI
  - confusion matrix at threshold 0.5
  - precision, recall and F1
  - permutation importance

## Status

Synthetic demonstration, version 0.1.0. No license has been assigned yet.
