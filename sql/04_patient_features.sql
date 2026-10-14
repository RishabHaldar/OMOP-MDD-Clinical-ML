-- 04_patient_features.sql
--
-- NEWLY RECONSTRUCTED EXAMPLE (2026). This is NOT the original WashU Databricks
-- query. See sql/README.md.
--
-- One row per cohort member: age at index, sex, the per-person median of each
-- of the 16 analytes (canonical units), and two completeness measures:
--   n_measurements      = total qualifying measurements (used for selection)
--   n_distinct_analytes = analytes with >= 1 qualifying measurement (descriptive)
-- Neither completeness measure is a model predictor.
--
-- Sex: gender_concept_id 8507 -> 1, 8532 -> 0, anything else -> NULL (missing).

CREATE OR REPLACE TABLE patient_features AS
WITH cohort AS (
    SELECT person_id, cohort, index_date, age_at_index, gender_concept_id
    FROM case_cohort
    UNION ALL
    SELECT person_id, cohort, index_date, age_at_index, gender_concept_id
    FROM comparison_cohort
),
labs AS (
    SELECT
        person_id,
        COUNT(*) AS n_measurements,
        COUNT(DISTINCT analyte) AS n_distinct_analytes,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'leukocytes') AS leukocytes_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'neutrophils') AS neutrophils_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'lymphocytes') AS lymphocytes_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'monocytes') AS monocytes_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'eosinophils') AS eosinophils_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'basophils') AS basophils_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'platelets') AS platelets_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'erythrocyte_width') AS erythrocyte_width_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'albumin') AS albumin_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'glucose') AS glucose_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'bmi') AS bmi_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'triglycerides') AS triglycerides_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'cholesterol') AS cholesterol_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'hdl') AS hdl_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'ldl') AS ldl_median,
        MEDIAN(value_canonical) FILTER (WHERE analyte = 'hba1c') AS hba1c_median
    FROM lab_measurements
    GROUP BY person_id
)
SELECT
    c.person_id,
    c.cohort,
    c.index_date,
    c.gender_concept_id,
    c.age_at_index AS age,
    CASE c.gender_concept_id
        WHEN 8507 THEN 1
        WHEN 8532 THEN 0
        ELSE NULL
    END AS is_male,
    COALESCE(l.n_measurements, 0) AS n_measurements,
    COALESCE(l.n_distinct_analytes, 0) AS n_distinct_analytes,
    l.leukocytes_median,
    l.neutrophils_median,
    l.lymphocytes_median,
    l.monocytes_median,
    l.eosinophils_median,
    l.basophils_median,
    l.platelets_median,
    l.erythrocyte_width_median,
    l.albumin_median,
    l.glucose_median,
    l.bmi_median,
    l.triglycerides_median,
    l.cholesterol_median,
    l.hdl_median,
    l.ldl_median,
    l.hba1c_median
FROM cohort AS c
LEFT JOIN labs AS l
    ON l.person_id = c.person_id;
