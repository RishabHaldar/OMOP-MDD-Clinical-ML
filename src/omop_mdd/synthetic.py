"""Independently generated synthetic OMOP CDM tables.

Nothing in this module is derived from real patient data. Laboratory values are
drawn from approximate textbook adult reference intervals chosen for
illustration only. Missingness arises from configurable panel-ordering
probabilities chosen for this demonstration, not from any institutional extract.
All clinical concept IDs are synthetic (see config/concept_sets.yaml).

Two independent random streams are used:

* the *record* stream generates persons, visits, observation periods and
  measurements;
* the *label* stream decides who is a case and generates condition records.

With ``signal == 0`` the record stream never reads case status, so person,
visit, observation-period and measurement tables are identical whichever
persons are cases. ``tests/test_synthetic.py`` checks this directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from omop_mdd import config

# Per-patient baseline distribution in the canonical unit: (mean, sd, minimum).
# Approximate textbook adult reference intervals, for illustration only.
LAB_DISTRIBUTIONS: dict[str, tuple[float, float, float]] = {
    "leukocytes": (7.5, 2.0, 1.0),  # 10^9/L
    "neutrophils": (4.5, 1.5, 0.3),  # 10^9/L
    "lymphocytes": (2.0, 0.7, 0.2),  # 10^9/L
    "monocytes": (0.6, 0.2, 0.05),  # 10^9/L
    "eosinophils": (0.2, 0.1, 0.0),  # 10^9/L
    "basophils": (0.05, 0.03, 0.0),  # 10^9/L
    "platelets": (270.0, 60.0, 50.0),  # 10^9/L
    "erythrocyte_width": (13.5, 1.2, 10.0),  # %
    "albumin": (4.0, 0.4, 2.0),  # g/dL
    "glucose": (100.0, 20.0, 50.0),  # mg/dL
    "bmi": (28.0, 6.0, 15.0),  # kg/m2
    "triglycerides": (130.0, 60.0, 30.0),  # mg/dL
    "cholesterol": (185.0, 35.0, 80.0),  # mg/dL
    "hdl": (50.0, 13.0, 15.0),  # mg/dL
    "ldl": (105.0, 30.0, 20.0),  # mg/dL
    "hba1c": (5.6, 0.6, 4.0),  # %
}

# Visit-to-visit variation around each patient's baseline (coefficient of variation).
WITHIN_PATIENT_CV = 0.08

# Labs ordered together, and the probability each panel is ordered at a visit.
PANELS: dict[str, tuple[list[str], float]] = {
    "cbc": (
        [
            "leukocytes",
            "neutrophils",
            "lymphocytes",
            "monocytes",
            "eosinophils",
            "basophils",
            "platelets",
            "erythrocyte_width",
        ],
        0.45,
    ),
    "metabolic": (["glucose", "albumin"], 0.45),
    "lipid": (["triglycerides", "cholesterol", "hdl", "ldl"], 0.20),
    "hba1c": (["hba1c"], 0.15),
    "vitals": (["bmi"], 0.55),
}

# Fraction of measurements recorded in the alternate compatible unit, and with
# unit_concept_id 0 ("No matching concept"), which the SQL must exclude.
ALTERNATE_UNIT_FRACTION = 0.15
UNMAPPED_UNIT_FRACTION = 0.05

# Direction of the purely synthetic case/control shift, in baseline-SD units
# multiplied by ``signal``. It has no clinical meaning.
SIGNAL_DIRECTIONS: dict[str, float] = {
    "glucose": 1.0,
    "hba1c": 1.0,
    "bmi": 1.0,
    "hdl": -1.0,
    "leukocytes": 1.0,
}

# Synthetic condition concepts.
DEPRESSION_ROOT = 2000000100
DEPRESSION_DESCENDANTS = [2000000101, 2000000102]
OTHER_MOOD_DISORDER = 2000000110
UNRELATED_CONDITIONS = [2000000201, 2000000202, 2000000203, 2000000204]
UNMAPPED_UNIT = 0

FIRST_VISIT_START = pd.Timestamp("2012-01-01")
FIRST_VISIT_END = pd.Timestamp("2020-12-31")


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the synthetic population."""

    n_persons: int = 1000
    case_fraction: float = 0.5
    signal: float = 0.5
    seed: int = 2026
    label_seed: int | None = None  # defaults to a stream derived from ``seed``
    other_mood_fraction: float = 0.08
    unrelated_condition_rate: float = 0.6
    age_at_first_visit_range: tuple[float, float] = (16.0, 80.0)
    mean_extra_visits: float = 5.0
    mean_days_between_visits: float = 120.0
    gender_probabilities: dict[int, float] = field(
        default_factory=lambda: {8507: 0.48, 8532: 0.50, 0: 0.02}
    )


def _date(ts: pd.Timestamp | np.datetime64) -> pd.Timestamp:
    return pd.Timestamp(ts).normalize()


def _vocabulary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the concept and concept_ancestor tables."""
    cs = config.concept_sets()
    rows = [
        (DEPRESSION_ROOT, "SYNTHETIC: Depressive disorder", "Condition"),
        (2000000101, "SYNTHETIC: Depressive episode", "Condition"),
        (2000000102, "SYNTHETIC: Recurrent depressive disorder", "Condition"),
        (OTHER_MOOD_DISORDER, "SYNTHETIC: Other mood disorder", "Condition"),
    ]
    rows += [
        (cid, f"SYNTHETIC: Unrelated condition {i + 1}", "Condition")
        for i, cid in enumerate(UNRELATED_CONDITIONS)
    ]
    rows += [
        (spec["concept_id"], f"SYNTHETIC: {name}", "Measurement")
        for name, spec in cs["labs"].items()
    ]
    rows += [
        (cid, f"SYNTHETIC unit: {unit}", "Unit") for unit, cid in cs["units"].items()
    ]
    rows += [
        (cs["gender"]["male"], "MALE", "Gender"),
        (cs["gender"]["female"], "FEMALE", "Gender"),
        (0, "No matching concept", "Metadata"),
    ]
    concept = pd.DataFrame(rows, columns=["concept_id", "concept_name", "domain_id"])
    concept["vocabulary_id"] = np.where(
        concept["concept_id"] >= 2_000_000_000, "SYNTHETIC", "Gender"
    )
    concept.loc[concept["concept_id"] == 0, "vocabulary_id"] = "None"
    concept["standard_concept"] = np.where(concept["concept_id"] == 0, None, "S")

    # Every concept is its own ancestor (OMOP convention), plus the depression
    # hierarchy.
    ancestors = [(cid, cid, 0) for cid in concept["concept_id"] if cid != 0]
    ancestors += [(DEPRESSION_ROOT, child, 1) for child in DEPRESSION_DESCENDANTS]
    concept_ancestor = pd.DataFrame(
        ancestors,
        columns=[
            "ancestor_concept_id",
            "descendant_concept_id",
            "min_levels_of_separation",
        ],
    )
    concept_ancestor["max_levels_of_separation"] = concept_ancestor[
        "min_levels_of_separation"
    ]
    return concept, concept_ancestor


def generate(cfg: SyntheticConfig | None = None) -> dict[str, pd.DataFrame]:
    """Generate synthetic OMOP tables.

    Returns a dict with keys: person, observation_period, visit_occurrence,
    condition_occurrence, measurement, concept, concept_ancestor. Also returns
    ``_truth`` (person_id, is_case) for testing only; it is never loaded into
    the database used by the SQL.
    """
    cfg = cfg or SyntheticConfig()
    cs = config.concept_sets()
    record_rng = np.random.default_rng([cfg.seed, 0])
    label_seed = cfg.label_seed if cfg.label_seed is not None else cfg.seed
    label_rng = np.random.default_rng([label_seed, 1])

    n = cfg.n_persons
    person_ids = np.arange(1, n + 1, dtype=np.int64)

    # --- Label stream: case status ------------------------------------------
    is_case = label_rng.random(n) < cfg.case_fraction

    # --- Record stream: persons and visits ----------------------------------
    genders = list(cfg.gender_probabilities.keys())
    gender_p = np.array(list(cfg.gender_probabilities.values()), dtype=float)
    gender = record_rng.choice(genders, size=n, p=gender_p / gender_p.sum())

    span_days = (FIRST_VISIT_END - FIRST_VISIT_START).days
    first_visit = FIRST_VISIT_START + pd.to_timedelta(
        record_rng.integers(0, span_days, size=n), unit="D"
    )
    lo, hi = cfg.age_at_first_visit_range
    age_days = (record_rng.uniform(lo, hi, size=n) * 365.25).astype(int)
    birth = first_visit - pd.to_timedelta(age_days, unit="D")

    person = pd.DataFrame(
        {
            "person_id": person_ids,
            "gender_concept_id": gender.astype(np.int64),
            "year_of_birth": birth.year,
            "month_of_birth": birth.month,
            "day_of_birth": birth.day,
        }
    )

    visit_rows = []
    visit_dates: list[list[pd.Timestamp]] = []
    visit_id = 1
    for i, pid in enumerate(person_ids):
        n_visits = 1 + record_rng.poisson(cfg.mean_extra_visits)
        gaps = record_rng.exponential(cfg.mean_days_between_visits, size=n_visits - 1)
        offsets = np.concatenate([[0.0], np.cumsum(np.maximum(gaps, 1.0))])
        dates = [_date(first_visit[i] + pd.Timedelta(days=float(o))) for o in offsets]
        visit_dates.append(dates)
        for d in dates:
            visit_rows.append((visit_id, pid, d, d))
            visit_id += 1
    visit_occurrence = pd.DataFrame(
        visit_rows,
        columns=[
            "visit_occurrence_id",
            "person_id",
            "visit_start_date",
            "visit_end_date",
        ],
    )

    observation_period = pd.DataFrame(
        {
            "observation_period_id": person_ids,
            "person_id": person_ids,
            "observation_period_start_date": [
                d[0] - pd.Timedelta(days=30) for d in visit_dates
            ],
            "observation_period_end_date": [
                d[-1] + pd.Timedelta(days=30) for d in visit_dates
            ],
        }
    )

    # --- Record stream: measurements ----------------------------------------
    analytes = config.analytes()
    baseline = np.empty((n, len(analytes)))
    for j, lab in enumerate(analytes):
        mean, sd, minimum = LAB_DISTRIBUTIONS[lab]
        baseline[:, j] = np.maximum(record_rng.normal(mean, sd, size=n), minimum)
    # The synthetic signal is added after all random draws, so it never changes
    # the record stream itself.
    if cfg.signal:
        for lab, direction in SIGNAL_DIRECTIONS.items():
            j = analytes.index(lab)
            sd = LAB_DISTRIBUTIONS[lab][1]
            baseline[is_case, j] += direction * cfg.signal * sd

    unit_ids = cs["units"]
    meas_rows = []
    meas_id = 1
    for i, pid in enumerate(person_ids):
        for d in visit_dates[i]:
            for labs, prob in PANELS.values():
                if record_rng.random() >= prob:
                    continue
                for lab in labs:
                    j = analytes.index(lab)
                    spec = cs["labs"][lab]
                    minimum = LAB_DISTRIBUTIONS[lab][2]
                    noise = record_rng.normal(0.0, WITHIN_PATIENT_CV)
                    value = max(baseline[i, j] * (1.0 + noise), minimum)
                    u = record_rng.random()
                    alt = spec["alternate_unit"]
                    if u < UNMAPPED_UNIT_FRACTION:
                        unit_id = UNMAPPED_UNIT
                        recorded = value
                    elif alt is not None and u < (
                        UNMAPPED_UNIT_FRACTION + ALTERNATE_UNIT_FRACTION
                    ):
                        unit_id = unit_ids[alt["unit"]]
                        recorded = (value - alt["offset"]) / alt["multiplier"]
                    else:
                        unit_id = unit_ids[spec["canonical_unit"]]
                        recorded = value
                    meas_rows.append(
                        (meas_id, pid, spec["concept_id"], d, recorded, unit_id)
                    )
                    meas_id += 1
    measurement = pd.DataFrame(
        meas_rows,
        columns=[
            "measurement_id",
            "person_id",
            "measurement_concept_id",
            "measurement_date",
            "value_as_number",
            "unit_concept_id",
        ],
    )

    # --- Label stream: conditions -------------------------------------------
    cond_rows = []
    cond_id = 1
    for i, pid in enumerate(person_ids):
        dates = visit_dates[i]
        if is_case[i]:
            # Diagnosis recorded at the first visit, so the case index date and
            # the comparison index date (first visit) follow the same
            # distribution. A repeat code at a later visit exercises MIN().
            concept_id = int(
                label_rng.choice([DEPRESSION_ROOT] + DEPRESSION_DESCENDANTS)
            )
            cond_rows.append((cond_id, pid, concept_id, dates[0]))
            cond_id += 1
            if len(dates) > 1 and label_rng.random() < 0.5:
                later = dates[int(label_rng.integers(1, len(dates)))]
                cond_rows.append((cond_id, pid, concept_id, later))
                cond_id += 1
        elif label_rng.random() < cfg.other_mood_fraction:
            d = dates[int(label_rng.integers(0, len(dates)))]
            cond_rows.append((cond_id, pid, OTHER_MOOD_DISORDER, d))
            cond_id += 1
        if label_rng.random() < cfg.unrelated_condition_rate:
            d = dates[int(label_rng.integers(0, len(dates)))]
            cid = int(label_rng.choice(UNRELATED_CONDITIONS))
            cond_rows.append((cond_id, pid, cid, d))
            cond_id += 1
    condition_occurrence = pd.DataFrame(
        cond_rows,
        columns=[
            "condition_occurrence_id",
            "person_id",
            "condition_concept_id",
            "condition_start_date",
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
        "_truth": pd.DataFrame({"person_id": person_ids, "is_case": is_case}),
    }
