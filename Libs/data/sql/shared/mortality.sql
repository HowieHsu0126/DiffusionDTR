-- ========================================================================
-- MORTALITY OUTCOME DATA EXTRACTION
-- Generates aki_90day_mortality.csv for AKI patient cohort
-- ========================================================================
\echo 'Generating AKI patient 90-day mortality outcomes...'
-- Drop temporary table if exists
DROP TABLE IF EXISTS temp_aki_90day_mortality;


/**
 * Extract 90-day mortality outcomes for AKI patients
 * 
 * This query computes mortality outcomes by:
 * 1. Identifying AKI patients from the final_aki_status table
 * 2. Determining death status from hospital admissions and patient records
 * 3. Calculating whether death occurred within 90 days of AKI diagnosis
 * 
 * Output columns:
 * - subject_id: Patient identifier
 * - death_within_90d: Binary flag (1=died within 90 days, 0=survived)
 * - death_date: Date of death (for validation, NULL if alive)
 * - earliest_aki_diagnosis_time: AKI diagnosis time (for validation)
 * - days_to_death: Number of days from AKI onset to death (NULL if alive)
 */
CREATE TEMP TABLE temp_aki_90day_mortality AS
WITH aki_patients AS (
    -- Get AKI patients with their onset time
    SELECT DISTINCT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        f.earliest_aki_diagnosis_time
    FROM
        final_aki_status f
    WHERE
        f.earliest_aki_diagnosis_time IS NOT NULL -- Only include patients with confirmed AKI diagnosis time
),
patient_death_info AS (
    -- Get patient death information from multiple sources
    SELECT
        p.subject_id,
        -- Use the earliest available death date
        COALESCE(p.dod, -- Date of death from patients table
            a.deathtime -- Death time from admissions table
) AS death_date,
        -- Determine if patient died during any hospitalization
        CASE WHEN p.dod IS NOT NULL
            OR a.deathtime IS NOT NULL THEN
            1
        ELSE
            0
        END AS died_flag
    FROM
        mimiciv_hosp.patients p
        LEFT JOIN mimiciv_hosp.admissions a ON p.subject_id = a.subject_id
    WHERE
        p.subject_id IN (
            SELECT
                subject_id
            FROM
                aki_patients)
),
-- Aggregate death info per patient (handle multiple admissions)
patient_death_final AS (
    SELECT
        subject_id,
        MAX(died_flag) AS died_flag,
        -- Use the earliest death date if multiple exist
        MIN(death_date) AS death_date
    FROM
        patient_death_info
    GROUP BY
        subject_id
),
-- Calculate 90-day mortality for each AKI patient
mortality_outcomes AS (
    SELECT
        a.subject_id,
        a.earliest_aki_diagnosis_time,
        d.death_date,
        d.died_flag,
        -- Calculate days from AKI onset to death
        CASE WHEN d.death_date IS NOT NULL
            AND a.earliest_aki_diagnosis_time IS NOT NULL THEN
            EXTRACT(DAY FROM d.death_date - a.earliest_aki_diagnosis_time)
        ELSE
            NULL
        END AS days_to_death,
        -- Determine 90-day mortality
        CASE WHEN d.died_flag = 1
            AND d.death_date IS NOT NULL
            AND a.earliest_aki_diagnosis_time IS NOT NULL
            AND EXTRACT(DAY FROM d.death_date - a.earliest_aki_diagnosis_time) <= 90 THEN
            1
        ELSE
            0
        END AS death_within_90d
    FROM
        aki_patients a
        LEFT JOIN patient_death_final d ON a.subject_id = d.subject_id
        -- Take the first AKI episode per patient for consistency
    WHERE
        a.earliest_aki_diagnosis_time =(
            SELECT
                MIN(earliest_aki_diagnosis_time)
            FROM
                aki_patients a2
            WHERE
                a2.subject_id = a.subject_id))
    -- Final selection with validation columns
    SELECT
        subject_id,
        death_within_90d,
        death_date,
        earliest_aki_diagnosis_time,
        days_to_death
    FROM
        mortality_outcomes
    ORDER BY
        subject_id;

-- Export mortality data to shared directory
\echo 'Exporting 90-day mortality data to CSV...'
\COPY (SELECT subject_id, death_within_90d FROM temp_aki_90day_mortality) TO '/home/hwxu/Projects/Research/PKU/AAAI2026/Input/raw/shared/aki_90day_mortality.csv' WITH CSV HEADER DELIMITER ',';

\echo '90-day mortality data generation completed!' 
