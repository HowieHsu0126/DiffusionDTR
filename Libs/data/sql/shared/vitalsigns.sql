-- ========================================================================
-- VITAL SIGNS TIME SERIES DATA EXTRACTION
-- Generates aki_vitalsigns_timeseries.csv for both vent/ and rrt/ directories
-- Uses HOURS_BEFORE_AKI and HOURS_AFTER_AKI variables
-- ========================================================================

\echo 'Generating shared vital signs data...'

-- 5. Vital Signs Time Series
DROP TABLE IF EXISTS temp_aki_vitalsigns_gcs_timeseries;

CREATE TEMP TABLE temp_aki_vitalsigns_gcs_timeseries AS
WITH first_icu AS (
    SELECT
        subject_id,
        MIN(hadm_id) AS hadm_id,
        MIN(stay_id) AS stay_id
    FROM
        final_aki_status
    GROUP BY
        subject_id
),
vital_window AS (
    SELECT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        f.earliest_aki_diagnosis_time,
        ROUND(EXTRACT(EPOCH FROM (v.charttime - f.earliest_aki_diagnosis_time)) / 3600) AS hours_from_aki,
        v.heart_rate,
        v.sbp,
        v.dbp,
        v.mbp,
        v.temperature,
        v.o2sat,
        v.resp_rate
    FROM
        final_aki_status f
        JOIN first_icu fi ON f.subject_id = fi.subject_id
            AND f.hadm_id = fi.hadm_id
            AND f.stay_id = fi.stay_id
        LEFT JOIN vitalsign v ON f.stay_id = v.stay_id
    WHERE
        f.earliest_aki_diagnosis_time IS NOT NULL
                AND v.charttime BETWEEN f.earliest_aki_diagnosis_time - (:HOURS_BEFORE_AKI || ' hours')::INTERVAL
            AND f.earliest_aki_diagnosis_time + (:HOURS_AFTER_AKI || ' hours')::INTERVAL
),
gcs_window AS (
    SELECT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        ROUND(EXTRACT(EPOCH FROM (g.charttime - f.earliest_aki_diagnosis_time)) / 3600) AS hours_from_aki,
        g.gcs
    FROM
        final_aki_status f
        JOIN first_icu fi ON f.subject_id = fi.subject_id
            AND f.hadm_id = fi.hadm_id
            AND f.stay_id = fi.stay_id
        LEFT JOIN mimiciv_derived.gcs g ON f.stay_id = g.stay_id
    WHERE
        f.earliest_aki_diagnosis_time IS NOT NULL
                AND g.charttime BETWEEN f.earliest_aki_diagnosis_time - (:HOURS_BEFORE_AKI || ' hours')::INTERVAL
            AND f.earliest_aki_diagnosis_time + (:HOURS_AFTER_AKI || ' hours')::INTERVAL
),
all_times AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        hours_from_aki
    FROM
        vital_window
    UNION
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        hours_from_aki
    FROM
        gcs_window
),
merged AS (
    SELECT
        t.subject_id,
        t.hours_from_aki,
        vw.heart_rate,
        vw.sbp,
        vw.dbp,
        vw.mbp,
        vw.temperature,
        vw.o2sat,
        vw.resp_rate,
        gw.gcs,
        -- SIRS score: temp+hr+resp
        (COALESCE(
            CASE WHEN vw.temperature < 36.0 THEN
                1
            WHEN vw.temperature > 38.0 THEN
                1
            WHEN vw.temperature IS NULL THEN
                NULL
            ELSE
                0
            END, 0) + COALESCE(
            CASE WHEN vw.heart_rate > 90.0 THEN
                1
            WHEN vw.heart_rate IS NULL THEN
                NULL
            ELSE
                0
            END, 0) + COALESCE(
            CASE WHEN vw.resp_rate > 20.0 THEN
                1
            WHEN vw.resp_rate IS NULL THEN
                NULL
            ELSE
                0
            END, 0)) AS sirs_score,
        ROW_NUMBER() OVER (PARTITION BY t.subject_id,
            t.hours_from_aki ORDER BY 
            (CASE WHEN vw.heart_rate IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN vw.sbp IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN vw.dbp IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN vw.mbp IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN vw.temperature IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN vw.o2sat IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN gw.gcs IS NOT NULL THEN
                0
            ELSE
                1
            END)) AS rn
    FROM
        all_times t
        LEFT JOIN vital_window vw ON t.subject_id = vw.subject_id
            AND t.hadm_id = vw.hadm_id
            AND t.stay_id = vw.stay_id
            AND t.hours_from_aki = vw.hours_from_aki
        LEFT JOIN gcs_window gw ON t.subject_id = gw.subject_id
            AND t.hadm_id = gw.hadm_id
            AND t.stay_id = gw.stay_id
            AND t.hours_from_aki = gw.hours_from_aki
    WHERE
        t.hours_from_aki BETWEEN -:HOURS_BEFORE_AKI AND :HOURS_AFTER_AKI
)
SELECT
    subject_id,
    hours_from_aki AS hours_from_onset,
    ROUND(CAST(gcs AS numeric), 2) AS gcs,
    ROUND(CAST(heart_rate AS numeric), 2) AS heart_rate,
    ROUND(CAST(sirs_score AS numeric), 0) AS sirs_score,
    ROUND(CAST(sbp AS numeric), 2) AS sbp,
    ROUND(CAST(dbp AS numeric), 2) AS dbp,
    ROUND(CAST(mbp AS numeric), 2) AS mbp,
    ROUND(CAST(temperature AS numeric), 2) AS temperature,
    ROUND(CAST(o2sat AS numeric), 2) AS spo2
FROM
    merged
WHERE
    rn = 1
ORDER BY
    subject_id,
    hours_from_onset;

-- Export vital signs to shared directory
\COPY (SELECT * FROM temp_aki_vitalsigns_gcs_timeseries) TO '/home/hwxu/Projects/Research/PKU/AAAI2026/Input/raw/shared/aki_vitalsigns_timeseries.csv' WITH CSV HEADER DELIMITER ','; 