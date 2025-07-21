-- ========================================================================
-- LABORATORY RESULTS TIME SERIES DATA EXTRACTION
-- Generates aki_labres_timeseries.csv for both vent/ and rrt/ directories  
-- Uses HOURS_BEFORE_AKI and HOURS_AFTER_AKI variables
-- ========================================================================

\echo 'Generating shared laboratory results...'

-- Create vitalsign table first (needed for CBC data)
DROP TABLE IF EXISTS vitalsign;

CREATE TABLE vitalsign AS
SELECT
    ce.subject_id,
    ce.stay_id,
    ce.charttime,
    AVG(
        CASE WHEN itemid = 220045
            AND valuenum > 0
            AND valuenum < 300 THEN
            valuenum
        ELSE
            NULL
        END) AS heart_rate,
    AVG(
        CASE WHEN itemid IN (220179, 220050)
            AND valuenum > 0
            AND valuenum < 400 THEN
            valuenum
        ELSE
            NULL
        END) AS sbp,
    AVG(
        CASE WHEN itemid IN (220180, 220051)
            AND valuenum > 0
            AND valuenum < 300 THEN
            valuenum
        ELSE
            NULL
        END) AS dbp,
    AVG(
        CASE WHEN itemid IN (220052, 220181, 225312)
            AND valuenum > 0
            AND valuenum < 300 THEN
            valuenum
        ELSE
            NULL
        END) AS mbp,
    AVG(
        CASE WHEN itemid IN (220210, 224690)
            AND valuenum > 0
            AND valuenum < 70 THEN
            valuenum
        ELSE
            NULL
        END) AS resp_rate,
    AVG(
        CASE WHEN itemid = 223761
            AND valuenum > 70
            AND valuenum < 120 THEN
            (valuenum - 32) / 1.8
        WHEN itemid = 223762
            AND valuenum > 10
            AND valuenum < 50 THEN
            valuenum
        ELSE
            NULL
        END) AS temperature,
    AVG(
        CASE WHEN itemid IN (220277, 50817)
            AND valuenum > 0
            AND valuenum <= 100 THEN
            valuenum
        ELSE
            NULL
        END) AS o2sat
FROM
    mimiciv_icu.chartevents ce
WHERE
    ce.stay_id IS NOT NULL
    AND ce.itemid IN (220045, -- Heart Rate
        220050, 220051, 220052, -- Arterial BP
        220179, 220180, 220181, -- NIBP
        225312, -- ART BP Mean
        220210, 224690, -- Resp Rate
        220277, 50817, -- SpO2
        223762, 223761 -- Temperature
)
GROUP BY
    ce.subject_id,
    ce.stay_id,
    ce.charttime;

-- 4. Laboratory Results Time Series
DROP TABLE IF EXISTS temp_aki_labres_timeseries;

CREATE TEMP TABLE temp_aki_labres_timeseries AS
WITH
-- 新增CTE：first_icu
first_icu AS (
    SELECT
        subject_id,
        MIN(hadm_id) AS hadm_id,
        MIN(stay_id) AS stay_id
    FROM
        final_aki_status
    GROUP BY
        subject_id
),
-- 尿量
uo_data AS (
    SELECT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        ROUND(EXTRACT(EPOCH FROM (uo.charttime - f.earliest_aki_diagnosis_time)) / 3600) AS hours_from_aki,
        uo.urineoutput
    FROM
        final_aki_status f
        JOIN first_icu fi ON f.subject_id = fi.subject_id
            AND f.hadm_id = fi.hadm_id
            AND f.stay_id = fi.stay_id
        LEFT JOIN mimiciv_derived.urine_output uo ON f.stay_id = uo.stay_id
    WHERE
        f.earliest_aki_diagnosis_time IS NOT NULL
                AND uo.charttime BETWEEN f.earliest_aki_diagnosis_time - (:HOURS_BEFORE_AKI || ' hours')::INTERVAL
            AND f.earliest_aki_diagnosis_time + (:HOURS_AFTER_AKI || ' hours')::INTERVAL
),
-- 肌酐
cr_data AS (
    SELECT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        ROUND(EXTRACT(EPOCH FROM (le.charttime - f.earliest_aki_diagnosis_time)) / 3600) AS hours_from_aki,
        le.valuenum AS creatinine
    FROM
        final_aki_status f
        JOIN first_icu fi ON f.subject_id = fi.subject_id
            AND f.hadm_id = fi.hadm_id
            AND f.stay_id = fi.stay_id
        LEFT JOIN mimiciv_hosp.labevents le ON f.subject_id = le.subject_id
            AND le.itemid = 50912
            AND le.valuenum IS NOT NULL
            AND le.valuenum <= 150
    WHERE
        f.earliest_aki_diagnosis_time IS NOT NULL
        AND le.charttime BETWEEN f.earliest_aki_diagnosis_time - (:HOURS_BEFORE_AKI || ' hours')::INTERVAL
        AND f.earliest_aki_diagnosis_time + (:HOURS_AFTER_AKI || ' hours')::INTERVAL
),
-- 其他实验室项目
other_data AS (
    SELECT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        ROUND(EXTRACT(EPOCH FROM (le.charttime - f.earliest_aki_diagnosis_time)) / 3600) AS hours_from_aki,
        MAX(
            CASE WHEN le.itemid = 50822 THEN
                le.valuenum
            END) AS potassium,
        MAX(
            CASE WHEN le.itemid = 50893 THEN
                le.valuenum
            END) AS calcium,
        MAX(
            CASE WHEN le.itemid = 50902 THEN
                le.valuenum
            END) AS chloride,
        MAX(
            CASE WHEN le.itemid = 51006 THEN
                le.valuenum
            END) AS bun,
        MAX(
            CASE WHEN le.itemid = 50983 THEN
                le.valuenum
            END) AS sodium,
        MAX(
            CASE WHEN le.itemid = 50931 THEN
                le.valuenum
            END) AS glucose,
        MAX(
            CASE WHEN le.itemid = 50930 THEN
                le.valuenum
            END) AS albumin,
        MAX(
            CASE WHEN le.itemid = 50976 THEN
                le.valuenum
            END) AS globulin,
        MAX(
            CASE WHEN le.itemid = 50813 THEN
                le.valuenum
            END) AS lactate,
        MAX(
            CASE WHEN le.itemid = 51274 THEN
                le.valuenum
            END) AS pt,
        MAX(
            CASE WHEN le.itemid = 51275 THEN
                le.valuenum
            END) AS ptt,
        MAX(
            CASE WHEN le.itemid = 51237 THEN
                le.valuenum
            END) AS inr,
        MAX(
            CASE WHEN le.itemid = 50802 THEN
                le.valuenum
            END) AS base_excess,
        MAX(
            CASE WHEN le.itemid = 50803 THEN
                le.valuenum
            END) AS bicarb
    FROM
        final_aki_status f
        JOIN first_icu fi ON f.subject_id = fi.subject_id
            AND f.hadm_id = fi.hadm_id
            AND f.stay_id = fi.stay_id
        LEFT JOIN mimiciv_hosp.labevents le ON f.subject_id = le.subject_id
            AND le.itemid IN (50822, 50893, 50902, 51006, 50983, 50931, 50930, 50976, 50813, 51274, 51275, 51237, 50802, 50803)
            AND le.valuenum IS NOT NULL
            AND (
                -- Potassium: 2.5-5.5 mEq/L (allow wider range for ICU patients)
                (le.itemid = 50822 AND le.valuenum BETWEEN 1.5 AND 8.0) OR
                -- Lactate: 0.5-2.0 mmol/L (allow wider range for ICU patients)  
                (le.itemid = 50813 AND le.valuenum BETWEEN 0.1 AND 20.0) OR
                -- Other lab values: use existing > 0 filter
                (le.itemid NOT IN (50822, 50813) AND le.valuenum > 0)
            )
    WHERE
        f.earliest_aki_diagnosis_time IS NOT NULL
        AND le.charttime BETWEEN f.earliest_aki_diagnosis_time - (:HOURS_BEFORE_AKI || ' hours')::INTERVAL
        AND f.earliest_aki_diagnosis_time + (:HOURS_AFTER_AKI || ' hours')::INTERVAL
    GROUP BY
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        hours_from_aki
),
-- 血常规
cbc_data AS (
    SELECT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        ROUND(EXTRACT(EPOCH FROM (cbc.charttime - f.earliest_aki_diagnosis_time)) / 3600) AS hours_from_aki,
        cbc.hemoglobin,
        cbc.platelet,
        cbc.wbc
    FROM
        final_aki_status f
        JOIN first_icu fi ON f.subject_id = fi.subject_id
            AND f.hadm_id = fi.hadm_id
            AND f.stay_id = fi.stay_id
        LEFT JOIN complete_blood_count cbc ON f.subject_id = cbc.subject_id
            AND cbc.charttime BETWEEN f.earliest_aki_diagnosis_time - (:HOURS_BEFORE_AKI || ' hours')::INTERVAL
            AND f.earliest_aki_diagnosis_time + (:HOURS_AFTER_AKI || ' hours')::INTERVAL
),
-- 合并所有时间点
all_times AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        hours_from_aki
    FROM
        uo_data
    UNION
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        hours_from_aki
    FROM
        cr_data
    UNION
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        hours_from_aki
    FROM
        other_data
    UNION
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        hours_from_aki
    FROM
        cbc_data
),
merged AS (
    SELECT
        t.subject_id,
        t.hadm_id,
        t.stay_id,
        t.hours_from_aki,
        uo.urineoutput,
        cr.creatinine,
        od.potassium,
        od.calcium,
        od.chloride,
        od.bun,
        od.sodium,
        od.glucose,
        od.albumin,
        od.globulin,
        od.lactate,
        od.pt,
        od.ptt,
        od.inr,
        od.base_excess,
        od.bicarb,
        cbc.hemoglobin,
        cbc.platelet,
        cbc.wbc,
        ROW_NUMBER() OVER (PARTITION BY t.subject_id,
            t.hours_from_aki ORDER BY
            -- 优先所有指标都非空的记录
            (CASE WHEN uo.urineoutput IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN cr.creatinine IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.potassium IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.calcium IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.chloride IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.bun IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.sodium IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.glucose IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.albumin IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.globulin IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.lactate IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.pt IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.ptt IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.inr IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.base_excess IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN od.bicarb IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN cbc.hemoglobin IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN cbc.platelet IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN cbc.wbc IS NOT NULL THEN
                0
            ELSE
                1
            END)
        ) AS rn
    FROM
        all_times t
        LEFT JOIN uo_data uo ON t.subject_id = uo.subject_id
            AND t.hadm_id = uo.hadm_id
            AND t.stay_id = uo.stay_id
            AND t.hours_from_aki = uo.hours_from_aki
        LEFT JOIN cr_data cr ON t.subject_id = cr.subject_id
            AND t.hadm_id = cr.hadm_id
            AND t.stay_id = cr.stay_id
            AND t.hours_from_aki = cr.hours_from_aki
        LEFT JOIN other_data od ON t.subject_id = od.subject_id
            AND t.hadm_id = od.hadm_id
            AND t.stay_id = od.stay_id
            AND t.hours_from_aki = od.hours_from_aki
        LEFT JOIN cbc_data cbc ON t.subject_id = cbc.subject_id
            AND t.hadm_id = cbc.hadm_id
            AND t.stay_id = cbc.stay_id
            AND t.hours_from_aki = cbc.hours_from_aki
    WHERE
        t.hours_from_aki BETWEEN -:HOURS_BEFORE_AKI AND :HOURS_AFTER_AKI
)
SELECT
    subject_id,
    hours_from_aki AS hours_from_onset,
    ROUND(CAST(urineoutput AS numeric), 2) AS urineoutput,
    ROUND(CAST(creatinine AS numeric), 2) AS creatinine,
    ROUND(CAST(potassium AS numeric), 2) AS potassium,
    ROUND(CAST(calcium AS numeric), 2) AS calcium,
    ROUND(CAST(chloride AS numeric), 2) AS chloride,
    ROUND(CAST(bun AS numeric), 2) AS bun,
    ROUND(CAST(sodium AS numeric), 2) AS sodium,
    ROUND(CAST(glucose AS numeric), 2) AS glucose,
    ROUND(CAST(albumin AS numeric), 2) AS albumin,
    ROUND(CAST(globulin AS numeric), 2) AS globulin,
    ROUND(CAST(lactate AS numeric), 2) AS lactate,
    ROUND(CAST(pt AS numeric), 2) AS pt,
    ROUND(CAST(ptt AS numeric), 2) AS ptt,
    ROUND(CAST(inr AS numeric), 2) AS inr,
    ROUND(CAST(base_excess AS numeric), 2) AS base_excess,
    ROUND(CAST(bicarb AS numeric), 2) AS bicarb,
    ROUND(CAST(hemoglobin AS numeric), 2) AS hemoglobin,
    ROUND(CAST(platelet AS numeric), 2) AS platelets,
    ROUND(CAST(wbc AS numeric), 2) AS wbc
FROM
    merged
WHERE
    rn = 1
ORDER BY
    subject_id,
    hours_from_aki;

-- Export laboratory results to shared directory
\COPY (SELECT * FROM temp_aki_labres_timeseries) TO '/home/hwxu/Projects/Research/PKU/AAAI2026/Input/raw/shared/aki_labres_timeseries.csv' WITH CSV HEADER DELIMITER ','; 