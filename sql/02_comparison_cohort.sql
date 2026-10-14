-- 02_comparison_cohort.sql
--
-- NEWLY RECONSTRUCTED EXAMPLE (2026). This is NOT the original WashU Databricks
-- query. See sql/README.md.
--
-- Comparison cohort: persons with at least one visit within an observation
-- period and NO condition record (at any time) in the `comparison_exclusion`
-- concept set, including descendants.
--   index date = first visit_start_date within an observation period
--   age        = completed years at index date (same rule as the case cohort)
--
-- The exclusion concept set is an open methodological decision (which
-- diagnoses count as "depression-related"). See sql/README.md.

CREATE OR REPLACE TABLE comparison_cohort AS
WITH excluded_concepts AS (
    SELECT DISTINCT ca.descendant_concept_id AS concept_id
    FROM concept_set_member AS csm
    JOIN concept_ancestor AS ca
        ON ca.ancestor_concept_id = csm.concept_id
    WHERE csm.concept_set = 'comparison_exclusion'
),
excluded_persons AS (
    SELECT DISTINCT co.person_id
    FROM condition_occurrence AS co
    JOIN excluded_concepts AS e
        ON co.condition_concept_id = e.concept_id
),
first_visit AS (
    SELECT
        v.person_id,
        MIN(v.visit_start_date) AS index_date
    FROM visit_occurrence AS v
    JOIN observation_period AS op
        ON op.person_id = v.person_id
        AND v.visit_start_date
            BETWEEN op.observation_period_start_date
            AND op.observation_period_end_date
    WHERE v.person_id NOT IN (SELECT person_id FROM excluded_persons)
    GROUP BY v.person_id
),
with_age AS (
    SELECT
        f.person_id,
        f.index_date,
        p.gender_concept_id,
        -- Age in completed years at the index date: year difference, minus
        -- one if the birthday has not yet been reached in the index year.
        (year(f.index_date) - p.year_of_birth)
        - CASE
            WHEN month(f.index_date) < p.month_of_birth
                OR (
                    month(f.index_date) = p.month_of_birth
                    AND day(f.index_date) < p.day_of_birth
                )
            THEN 1
            ELSE 0
        END AS age_at_index
    FROM first_visit AS f
    JOIN person AS p
        ON p.person_id = f.person_id
)
SELECT
    w.person_id,
    'comparison' AS cohort,
    w.index_date,
    w.age_at_index,
    w.gender_concept_id
FROM with_age AS w
CROSS JOIN analysis_params AS ap
WHERE w.age_at_index >= ap.min_age
    AND w.age_at_index <= ap.max_age
    AND w.person_id NOT IN (SELECT person_id FROM case_cohort);
