-- ========================================================================
-- ENHANCED IV FLUIDS AND VASOPRESSOR EXTRACTION FOR DTR OPTIMIZATION
-- ========================================================================
-- 
-- Extracts comprehensive IV fluids and vasopressor data with 5-level binning:
-- - IV Fluids: 0=none, 1=(0,50], 2=(50,180], 3=(180,530], 4=(530,∞) ml/hr
-- - Vasopressor: 0=none, 1=(0,0.08], 2=(0.08,0.22], 3=(0.22,0.45], 4=(0.45,∞) μg/kg/min
--
-- Data Sources:
-- - IV Fluids: mimiciv_icu.inputevents (crystalloids)
-- - Vasopressor: mimiciv_icu.inputevents (norepinephrine, epinephrine, etc.)
-- ========================================================================

-- Create enhanced IV fluids and vasopressor time series
DROP TABLE IF EXISTS temp_aki_ivfluids_vasopressor;

CREATE TEMP TABLE temp_aki_ivfluids_vasopressor AS
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

-- Get patient weight for vasopressor dosage calculation (μg/kg/min)
patient_weight AS (
    SELECT
        stay_id,
        subject_id,
        AVG(weight) AS weight_kg
    FROM (
        WITH wt_stg AS (
            SELECT
                c.stay_id,
                c.subject_id,
                c.charttime,
                CASE WHEN c.itemid = 226512 THEN 'admit'
                     ELSE 'daily'
                END AS weight_type,
                c.valuenum AS weight
            FROM
                mimiciv_icu.chartevents c
            WHERE
                c.valuenum IS NOT NULL
                AND c.itemid IN (226512, 224639) -- Admit Weight, Daily Weight
                AND c.valuenum > 0
        )
        SELECT DISTINCT
            stay_id,
            subject_id,
            weight
        FROM
            wt_stg
        WHERE
            weight IS NOT NULL
    ) w
    GROUP BY
        stay_id,
        subject_id
),

-- Extract IV fluids data (crystalloids)
iv_fluids_raw AS (
    SELECT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        ie.starttime,
        ie.endtime,
        ie.rate AS fluid_rate_ml_hr,
        ie.amount AS fluid_amount_ml,
        f.earliest_aki_diagnosis_time,
        ROUND(EXTRACT(EPOCH FROM (ie.starttime - f.earliest_aki_diagnosis_time)) / 3600) AS hours_from_onset
    FROM
        final_aki_status f
        JOIN first_icu fi ON f.subject_id = fi.subject_id
            AND f.hadm_id = fi.hadm_id
            AND f.stay_id = fi.stay_id
        LEFT JOIN mimiciv_icu.inputevents ie ON f.stay_id = ie.stay_id
    WHERE
        f.earliest_aki_diagnosis_time IS NOT NULL
        AND ie.itemid IN (
            225158, -- NaCl 0.9%
            225828, -- LR (Lactated Ringers)
            225823, -- D5 1/2NS
            226364, -- OR Crystalloid Intake
            225161, -- NaCl 3% (Hypertonic Saline)
            225825, -- D5NS
            225827, -- D5LR
            226375, -- PACU Crystalloid Intake
            225941  -- D5 1/4NS
        )
        AND ie.starttime BETWEEN f.earliest_aki_diagnosis_time - (:HOURS_BEFORE_AKI || ' hours')::INTERVAL
            AND f.earliest_aki_diagnosis_time + (:HOURS_AFTER_AKI || ' hours')::INTERVAL
        AND ie.rate IS NOT NULL
        AND ie.rate > 0
),

-- Extract vasopressor data (comprehensive vasopressors)
vasopressor_raw AS (
    SELECT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        ie.starttime,
        ie.endtime,
        ie.rate AS vaso_rate_mcg_min,
        ie.amount AS vaso_amount,
        pw.weight_kg,
        -- Convert to μg/kg/min
        CASE WHEN pw.weight_kg IS NOT NULL AND pw.weight_kg > 0 THEN
            ROUND(CAST(ie.rate / pw.weight_kg AS numeric), 3)
        ELSE
            NULL
        END AS vaso_rate_mcg_kg_min,
        f.earliest_aki_diagnosis_time,
        ROUND(EXTRACT(EPOCH FROM (ie.starttime - f.earliest_aki_diagnosis_time)) / 3600) AS hours_from_onset
    FROM
        final_aki_status f
        JOIN first_icu fi ON f.subject_id = fi.subject_id
            AND f.hadm_id = fi.hadm_id
            AND f.stay_id = fi.stay_id
        LEFT JOIN mimiciv_icu.inputevents ie ON f.stay_id = ie.stay_id
        LEFT JOIN patient_weight pw ON f.stay_id = pw.stay_id
    WHERE
        f.earliest_aki_diagnosis_time IS NOT NULL
        AND ie.itemid IN (
            221906, -- Norepinephrine (most common)
            221749, -- Phenylephrine
            229630, -- Phenylephrine (50/250)
            222315, -- Vasopressin
            221289, -- Epinephrine
            221662, -- Dopamine
            221653, -- Dobutamine
            229632, -- Phenylephrine (200/250)
            229617  -- Epinephrine.
        )
        AND ie.starttime BETWEEN f.earliest_aki_diagnosis_time - (:HOURS_BEFORE_AKI || ' hours')::INTERVAL
            AND f.earliest_aki_diagnosis_time + (:HOURS_AFTER_AKI || ' hours')::INTERVAL
        AND ie.rate IS NOT NULL
        AND ie.rate > 0
),

-- Aggregate by hourly windows
hourly_aggregated AS (
    SELECT
        subject_id,
        hours_from_onset,
        -- IV Fluids aggregation (sum all rates per hour)
        COALESCE(SUM(fluid_rate_ml_hr), 0) AS total_fluid_rate_ml_hr,
        -- Vasopressor aggregation (max rate per hour - most clinically relevant)
        MAX(vaso_rate_mcg_kg_min) AS max_vaso_rate_mcg_kg_min
    FROM (
        -- Union IV fluids and vasopressor data
        SELECT
            subject_id,
            hours_from_onset,
            fluid_rate_ml_hr,
            NULL::numeric AS vaso_rate_mcg_kg_min
        FROM
            iv_fluids_raw
        
        UNION ALL
        
        SELECT
            subject_id,
            hours_from_onset,
            NULL::numeric AS fluid_rate_ml_hr,
            vaso_rate_mcg_kg_min
        FROM
            vasopressor_raw
    ) combined
    GROUP BY
        subject_id,
        hours_from_onset
),

-- Apply binning logic
binned_data AS (
    SELECT
        subject_id,
        hours_from_onset,
        total_fluid_rate_ml_hr,
        max_vaso_rate_mcg_kg_min,
        
        -- IV Fluids binning (ml/hr)
        CASE 
            WHEN total_fluid_rate_ml_hr IS NULL OR total_fluid_rate_ml_hr = 0 THEN 0
            WHEN total_fluid_rate_ml_hr > 0 AND total_fluid_rate_ml_hr <= 50 THEN 1
            WHEN total_fluid_rate_ml_hr > 50 AND total_fluid_rate_ml_hr <= 180 THEN 2
            WHEN total_fluid_rate_ml_hr > 180 AND total_fluid_rate_ml_hr <= 530 THEN 3
            WHEN total_fluid_rate_ml_hr > 530 THEN 4
            ELSE 0
        END AS iv_fluids_bin,
        
        -- Vasopressor binning (μg/kg/min)
        CASE 
            WHEN max_vaso_rate_mcg_kg_min IS NULL OR max_vaso_rate_mcg_kg_min = 0 THEN 0
            WHEN max_vaso_rate_mcg_kg_min > 0 AND max_vaso_rate_mcg_kg_min <= 0.08 THEN 1
            WHEN max_vaso_rate_mcg_kg_min > 0.08 AND max_vaso_rate_mcg_kg_min <= 0.22 THEN 2
            WHEN max_vaso_rate_mcg_kg_min > 0.22 AND max_vaso_rate_mcg_kg_min <= 0.45 THEN 3
            WHEN max_vaso_rate_mcg_kg_min > 0.45 THEN 4
            ELSE 0
        END AS vasopressor_bin
    FROM
        hourly_aggregated
),

-- Generate complete time series (fill missing hours with 0)
complete_time_series AS (
    SELECT
        f.subject_id,
        generate_series(
            -:HOURS_BEFORE_AKI,
            :HOURS_AFTER_AKI,
            1
        ) AS hours_from_onset
    FROM
        final_aki_status f
        JOIN first_icu fi ON f.subject_id = fi.subject_id
            AND f.hadm_id = fi.hadm_id
            AND f.stay_id = fi.stay_id
    WHERE
        f.earliest_aki_diagnosis_time IS NOT NULL
)

-- Final SELECT for table creation (only discretized action space)
SELECT
    cts.subject_id,
    cts.hours_from_onset,
    COALESCE(bd.iv_fluids_bin, 0) AS iv_fluids_bin,
    COALESCE(bd.vasopressor_bin, 0) AS vasopressor_bin
FROM
    complete_time_series cts
    LEFT JOIN binned_data bd ON cts.subject_id = bd.subject_id
        AND cts.hours_from_onset = bd.hours_from_onset
ORDER BY
    cts.subject_id,
    cts.hours_from_onset;

-- Export IV fluids and vasopressor data to task directory
\COPY (SELECT * FROM temp_aki_ivfluids_vasopressor) TO '/home/hwxu/Projects/Research/PKU/AAAI2026/Input/raw/task/aki_ivfluids_vasopressor.csv' WITH CSV HEADER DELIMITER ','; 