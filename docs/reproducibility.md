# Reproducibility

## Environment

Developed and tested with **Python 3.13** on Linux. `requirements.txt` lists the exact versions used:

| Package | Version |
|---|---|
| numpy | 2.5.4 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| lightgbm | 4.7.0 |
| duckdb | 1.5.6 |
| PyYAML | 6.0.3 |
| matplotlib | 3.11.2 |
| pytest | 9.1.1 |
| black | 26.10.1 |

Other Python versions and operating systems have not been tested. LightGBM on macOS may need OpenMP (`brew install libomp`).

## Setup

From the repository directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

A conda environment works the same way: create it with Python 3.13, then run the two `pip install` lines.

## Commands

| Command | What it does | Typical time |
|---|---|---|
| `python -m omop_mdd.run --seed 2026` | Full synthetic demonstration; writes `outputs/` | ~30 s |
| `python -m omop_mdd.run --seed 2026 --signal 0` | Same, with no synthetic case/control signal | ~30 s |
| `pytest` | Test suite | ~1 min |
| `black --check .` | Formatting check | seconds |
| `python -m omop_mdd.null_check --reps 30` | Repeated null simulations (optional sanity check) | ~4 min |

## Pipeline

1. `omop_mdd.synthetic` generates OMOP-format tables: person, observation_period, visit_occurrence, condition_occurrence, measurement, concept, concept_ancestor.
2. `omop_mdd.extract` loads them into an in-memory DuckDB database and runs `sql/01`–`05`.
3. `omop_mdd.data` checks that person IDs are unique, cohorts don't overlap, and cohort sizes match. Labels come from the SQL `cohort` column.
4. `omop_mdd.features` restricts the data to the 18-predictor allowlist (`config/features.yaml`). An `AllowlistGuard` step re-checks the columns inside every Pipeline fit and predict.
5. `omop_mdd.evaluate`:
   - Stratified 75/25 patient-level split, seed-fixed.
   - 5-fold stratified cross-validation on the training set only.
   - One evaluation on the held-out set.
6. Both models share the same preprocessing, inside the Pipeline: a median `SimpleImputer` fitted on training rows only. Logistic regression adds `StandardScaler`.

## Outputs (`outputs/`, git-ignored)

| File | Contents |
|---|---|
| `metrics.json` | Split description, CV AUROC per fold, held-out metrics (AUROC with bootstrap 95% CI, confusion matrix at threshold 0.5, precision, recall, F1), per-cohort descriptive summary, package versions |
| `test_predictions.csv` | Synthetic person IDs, labels, predicted probabilities (held-out set) |
| `importance_<model>.csv` | Standardized coefficients (LR) or gain (LightGBM), plus permutation importance on the held-out set |
| `roc_test.png`, `confusion_test.png`, `permutation_importance.png`, `lr_coefficients.png` | Figures, each labelled as synthetic |

## Tests

| File | What it checks |
|---|---|
| `test_synthetic.py` | Determinism; synthetic concept IDs; with no signal, every non-condition table is identical whichever persons are cases |
| `test_sql.py` | The SQL runs; cohort sizes and no overlap; the same age-at-index rule in both cohorts; index-date definitions; measurement-count selection; unit conversion and exclusion; unknown gender → missing; lab window |
| `test_labels.py` | Label counts; labels match the SQL cohort and the generator's ground truth; duplicate, overlapping, unknown or mis-sized cohorts are rejected |
| `test_leakage.py` | The feature matrix equals the allowlist; `group`, `label`, `person_id`, `n_measurements` or any unknown column makes fit or predict fail |
| `test_preprocessing.py` | The split is patient-level; imputer statistics equal training-set medians (not pooled); preprocessing is unchanged when labels are shuffled; the imputer is refitted inside each CV fold |
| `test_determinism.py` | Same seed → identical results; different seed → different results |
| `test_outputs.py` | Files written; metrics, predictions and importance schemas |

The tests do not assert any particular AUROC. The no-signal behaviour is checked structurally in `test_synthetic.py` and, optionally, by repeated simulation with `omop_mdd.null_check`.

## Known limitations

- **Synthetic data only.** Distributions are illustrative textbook-style reference intervals. Missingness comes from arbitrary panel-ordering probabilities. The case/control signal is an artificial shift. Metrics from this repository describe the pipeline on synthetic data and say nothing about depression or about the historical study.
- **The original study cannot be reproduced from this repository.** The original Databricks SQL and observation windows are unavailable, and the institutional data is not included.
- **SQL dialect.** The queries run in DuckDB. Databricks equivalents are listed in `sql/README.md` but have not been executed.
- **Small held-out set.** With 50 test patients, held-out metrics vary noticeably with the seed. The bootstrap confidence interval reflects this.
- **No hyperparameter tuning.** Both models use fixed settings.
- **CI.** The workflow runs `black --check` and `pytest` on Python 3.13. It has not yet run on GitHub.
