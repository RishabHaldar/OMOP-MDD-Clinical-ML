-- 05_completeness_selection.sql
--
-- NEWLY RECONSTRUCTED EXAMPLE (2026). This is NOT the original WashU Databricks
-- query. See sql/README.md.
--
-- Within each cohort, rank persons by the total number of qualifying laboratory
-- measurements (descending) and keep the top n_per_cohort. Ties are broken by
-- person_id (ascending) so the selection is deterministic.
--
-- This follows the measurement-count criterion described for the historical
-- study. n_distinct_analytes is carried through as a descriptive measure only.

CREATE OR REPLACE TABLE analysis_dataset AS
WITH ranked AS (
    SELECT
        pf.*,
        ROW_NUMBER() OVER (
            PARTITION BY pf.cohort
            ORDER BY pf.n_measurements DESC, pf.person_id ASC
        ) AS completeness_rank
    FROM patient_features AS pf
)
SELECT r.* EXCLUDE (completeness_rank)
FROM ranked AS r
CROSS JOIN analysis_params AS ap
WHERE r.completeness_rank <= ap.n_per_cohort
ORDER BY r.cohort, r.person_id;
