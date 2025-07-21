-- ========================================================================
-- SOFA SCORE TIME SERIES DATA EXTRACTION
-- Generates sofa_aki.csv for both vent/ and rrt/ directories
-- Uses HOURS_BEFORE_AKI and HOURS_AFTER_AKI variables
-- ========================================================================

\echo 'Generating shared SOFA score data...'

-- 3. SOFA Score Time Series
DROP TABLE IF EXISTS temp_aki_sofa_timeseries;

CREATE TEMP TABLE temp_aki_sofa_timeseries AS
WITH aki_cohort AS (
    SELECT
        f.subject_id,
        i.stay_id,
        i.intime,
        i.outtime,
        f.earliest_aki_diagnosis_time,
        (f.earliest_aki_diagnosis_time - (:HOURS_BEFORE_AKI || ' hours')::INTERVAL) AS window_start,
        (f.earliest_aki_diagnosis_time + (:HOURS_AFTER_AKI || ' hours')::INTERVAL) AS window_end
    FROM
        final_aki_status f
        JOIN mimiciv_icu.icustays i ON f.subject_id = i.subject_id
    WHERE
        f.earliest_aki_diagnosis_time BETWEEN i.intime AND i.outtime
),
hours_window AS (
    SELECT
        a.subject_id,
        a.stay_id,
        generate_series(a.window_start, a.window_end, INTERVAL '1 hour') AS hour_time,
        a.earliest_aki_diagnosis_time
    FROM
        aki_cohort a
),
hours_from_onset AS (
    SELECT
        subject_id,
        stay_id,
        hour_time,
        EXTRACT(EPOCH FROM (hour_time - earliest_aki_diagnosis_time)) / 3600 AS hours_from_onset
    FROM
        hours_window
),
sofa_scores AS (
    SELECT
        h.subject_id,
        h.hours_from_onset,
        s.sofa_24hours AS sofa_score
    FROM
        hours_from_onset h
        LEFT JOIN mimiciv_derived.sofa s ON h.stay_id = s.stay_id
            AND h.hour_time = s.endtime
)
SELECT
    subject_id,
    hours_from_onset,
    sofa_score
FROM
    sofa_scores
ORDER BY
    subject_id,
    hours_from_onset;

-- Export SOFA to shared directory
\COPY (SELECT * FROM temp_aki_sofa_timeseries) TO '/home/hwxu/Projects/Research/PKU/AAAI2026/Input/raw/shared/sofa_aki.csv' WITH CSV HEADER DELIMITER ','; 