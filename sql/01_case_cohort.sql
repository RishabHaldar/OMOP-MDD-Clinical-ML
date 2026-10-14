-- 01_case_cohort.sql
--
-- NEWLY RECONSTRUCTED EXAMPLE (2026). This is NOT the original WashU Databricks
-- query, which is unavailable. Written in DuckDB SQL against OMOP CDM v5.4
-- table names; see sql/README.md for Databricks differences.
--
-- Case cohort: persons with at least one condition in the `depressive_disorder`
-- concept set (including descendants), within an observation period.
--   index date = first qualifying condition_start_date
--   age        = age at index date (same rule as the comparison cohort)
--
-- Inputs prepared by omop_mdd.extract:
--   analysis_params(min_age, max_age, ...)
--   concept_set_member(concept_set, concept_id)

CREATE OR REPLACE TABLE case_cohort AS
WITH dx_concepts AS (
    SELECT DISTINCT ca.descendant_concept_id AS concept_id
    FROM concept_set_member AS csm
    JOIN concept_ancestor AS ca
        ON ca.ancestor_concept_id = csm.concept_id
    WHERE csm.concept_set = 'depressive_disorder'
),
first_dx AS (
    SELECT
        co.person_id,
        MIN(co.condition_start_date) AS index_date
    FROM condition_occurrence AS co
    JOIN dx_concepts AS d
        ON co.condition_concept_id = d.concept_id
    JOIN observation_period AS op
        ON op.person_id = co.person_id
        AND co.condition_start_date
            BETWEEN op.observation_period_start_date
            AND op.observation_period_end_date
    GROUP BY co.person_id
),
with_age AS (
    SELECT
        f.person_id,
        f.index_date,
        p.gender_concept_id,
        date_diff(
            'day',
            make_date(p.year_of_birth, p.month_of_birth, p.day_of_birth),
            f.index_date
        ) / 365.25 AS age_at_index
    FROM first_dx AS f
    JOIN person AS p
        ON p.person_id = f.person_id
)
SELECT
    w.person_id,
    'case' AS cohort,
    w.index_date,
    w.age_at_index,
    w.gender_concept_id
FROM with_age AS w
CROSS JOIN analysis_params AS ap
WHERE w.age_at_index >= ap.min_age
    AND w.age_at_index <= ap.max_age;
