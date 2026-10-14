-- 03_lab_measurements.sql
--
-- NEWLY RECONSTRUCTED EXAMPLE (2026). This is NOT the original WashU Databricks
-- query. See sql/README.md.
--
-- Qualifying laboratory measurements for persons in either cohort:
--   * measurement concept in an analyte concept set (including descendants);
--   * unit_concept_id is the analyte's canonical unit or a listed compatible
--     unit, converted to the canonical unit as value * multiplier + offset;
--     any other unit (including 0, "No matching concept") is excluded;
--   * non-null value, within the person's observation period;
--   * optionally within a window around the index date (NULL = unbounded).
--
-- Inputs prepared by omop_mdd.extract:
--   lab_concept(analyte, concept_id)
--   unit_conversion(analyte, unit_concept_id, multiplier, "offset")
--   analysis_params(window_days_before_index, window_days_after_index, ...)

CREATE OR REPLACE TABLE lab_measurements AS
WITH cohort AS (
    SELECT person_id, cohort, index_date FROM case_cohort
    UNION ALL
    SELECT person_id, cohort, index_date FROM comparison_cohort
),
analyte_concepts AS (
    SELECT DISTINCT
        lc.analyte,
        ca.descendant_concept_id AS concept_id
    FROM lab_concept AS lc
    JOIN concept_ancestor AS ca
        ON ca.ancestor_concept_id = lc.concept_id
)
SELECT
    m.person_id,
    c.cohort,
    ac.analyte,
    m.measurement_date,
    m.value_as_number * uc.multiplier + uc."offset" AS value_canonical
FROM measurement AS m
JOIN cohort AS c
    ON c.person_id = m.person_id
JOIN analyte_concepts AS ac
    ON ac.concept_id = m.measurement_concept_id
JOIN unit_conversion AS uc
    ON uc.analyte = ac.analyte
    AND uc.unit_concept_id = m.unit_concept_id
JOIN observation_period AS op
    ON op.person_id = m.person_id
    AND m.measurement_date
        BETWEEN op.observation_period_start_date
        AND op.observation_period_end_date
CROSS JOIN analysis_params AS ap
WHERE m.value_as_number IS NOT NULL
    AND (
        ap.window_days_before_index IS NULL
        OR m.measurement_date >= c.index_date - CAST(ap.window_days_before_index AS INTEGER)
    )
    AND (
        ap.window_days_after_index IS NULL
        OR m.measurement_date <= c.index_date + CAST(ap.window_days_after_index AS INTEGER)
    );
