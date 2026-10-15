# OMOP-MDD-Clinical-ML

**Retrospective depression-phenotype classification with structured EHR data, SQL, and machine learning.**

This repository grew out of a graduate biomedical informatics project at Washington University in St. Louis examining associations between routinely collected laboratory measurements and recorded depressive-disorder diagnoses. The original analysis used OMOP-mapped electronic health record data accessed through Databricks SQL.

The code published here is a **separate, fully synthetic reconstruction** of that workflow. It demonstrates cohort extraction, longitudinal laboratory aggregation, and leakage-aware model evaluation without distributing institutional patient records or presenting synthetic model performance as a clinical finding.

## Research background

The original project assembled diagnosis-based and comparison cohorts, selecting 100 patients from each based on the availability of laboratory measurements. Repeated observations were summarized as patient-level medians for 16 laboratory features, alongside age and a demographic variable derived from recorded gender. The analysis explored gradient-boosted classifiers, including LightGBM, to distinguish recorded depressive-disorder cases from comparison patients.

The research question was exploratory: could structured laboratory and demographic variables help characterize differences associated with documented depression diagnoses in an EHR? Diagnosis codes were not equivalent to independently adjudicated major depressive disorder. The study was retrospective and did not establish whether laboratory features preceded disease onset.

A subsequent review identified limitations in cohort eligibility, laboratory-unit consistency, and preprocessing in the historical analysis. Its original performance estimates have not been revalidated and are not reproduced or claimed by this repository.

## Public demonstration

The implementation is independently written and runs end to end on generated data:

1. **Generate synthetic clinical records.** Python creates minimal OMOP-shaped `person`, `observation_period`, `visit_occurrence`, `condition_occurrence`, `measurement`, `concept`, and `concept_ancestor` tables. Synthetic concept identifiers and illustrative laboratory distributions are used throughout.
2. **Execute reconstructed SQL.** Five DuckDB queries build case and comparison cohorts, calculate age at a cohort-specific index date, validate and harmonize measurement units, calculate patient-level laboratory medians, and select patients by total qualifying measurement count.
3. **Evaluate two classifiers.** Logistic regression provides a baseline alongside LightGBM. An explicit allowlist limits modeling to age, `is_male`, and 16 laboratory medians. Imputation is fitted within training folds; evaluation uses five-fold cross-validation on the training set and a separate 25% held-out set.
4. **Inspect results.** The run writes aggregate metrics, model comparisons, and figures to a locally ignored `outputs/` directory. Automated tests check cohort integrity, age boundaries, unit handling, predictor leakage, and reproducibility.

**The synthetic signal is deliberately programmed into selected laboratory variables**—glucose, HbA1c, BMI, HDL, and leukocytes—to exercise the pipeline. Resulting feature importances and AUROCs reflect that construction; they are not evidence of biological associations with depression.

## Run locally

Developed and tested on Linux with Python 3.13. From the repository directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

python -m omop_mdd.run --seed 2026
pytest
black --check .
```

On Windows, activate the environment using the corresponding PowerShell or Command Prompt command. See [reproducibility notes](docs/reproducibility.md) for dependencies, generated outputs, and testing details.

## Repository guide

| Location | Contents |
|---|---|
| [`sql/`](sql/) | Five reconstructed cohort and feature-engineering queries; see the [SQL methodology notes](sql/README.md) |
| [`src/omop_mdd/synthetic.py`](src/omop_mdd/synthetic.py) | Independent synthetic record generator |
| [`src/omop_mdd/`](src/omop_mdd/) | Data extraction, feature validation, modeling, and evaluation |
| [`config/`](config/) | Synthetic concept mappings, laboratory units, cohort parameters, and predictor allowlist |
| [`tests/`](tests/) | Automated data-integrity, SQL, preprocessing, and model tests |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Setup, commands, evaluation details, and limitations |

## Interpretation and limitations

This is a **software and methods demonstration**, not a validated clinical model. The default laboratory observation window includes measurements after the index date; the pipeline therefore performs retrospective cohort classification and cannot predict incident depression. Case index dates are first qualifying diagnoses, while comparison index dates are first visits. Those dates are aligned by design in the synthetic generator but would require careful matching in a real study.

Selection by laboratory measurement count can favor patients with greater healthcare utilization. The generated tables include only the OMOP columns needed for this example and have not been validated as a full OMOP CDM implementation. DuckDB SQL has been executed against the synthetic data; suggested Databricks translations have not been tested. These choices and the open clinical concept-set decisions are detailed in [`sql/README.md`](sql/README.md).

No original WashU patient-level datasets, institutional SQL, or study-derived distributions are included. The repository does not reproduce the historical study's results.

## Status and use

Version 0.1.0. No open-source license has been assigned. Please contact the repository owner before reusing or redistributing the source code.
