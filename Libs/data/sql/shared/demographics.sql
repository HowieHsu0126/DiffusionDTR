-- ========================================================================
-- DEMOGRAPHICS DATA EXTRACTION
-- Generates aki_demographics.csv for both vent/ and rrt/ directories
-- ========================================================================

\echo 'Generating shared demographic data...'

-- 1. Demographics
DROP TABLE IF EXISTS temp_aki_demographics;

CREATE TEMP TABLE temp_aki_demographics AS
WITH
-- Height calculations
ht_in AS (
    SELECT
        c.subject_id,
        c.stay_id,
        c.charttime,
        ROUND(CAST((c.valuenum * 2.54) AS numeric), 2) AS height
    FROM
        mimiciv_icu.chartevents c
    WHERE
        c.valuenum IS NOT NULL
        AND c.itemid = 226707 -- Height (inches)
),
ht_cm AS (
    SELECT
        c.subject_id,
        c.stay_id,
        c.charttime,
        ROUND(CAST(c.valuenum AS numeric), 2) AS height
    FROM
        mimiciv_icu.chartevents c
    WHERE
        c.valuenum IS NOT NULL
        AND c.itemid = 226730 -- Height (cm)
),
height_merged AS (
    SELECT
        COALESCE(h1.subject_id, h2.subject_id) AS subject_id,
        COALESCE(h1.stay_id, h2.stay_id) AS stay_id,
        COALESCE(h1.height, h2.height) AS height
    FROM
        ht_cm h1
        FULL OUTER JOIN ht_in h2 ON h1.subject_id = h2.subject_id
        AND h1.stay_id = h2.stay_id
    WHERE
        COALESCE(h1.height, h2.height) > 120
        AND COALESCE(h1.height, h2.height) < 230
),
-- Weight calculations
weight_data AS (
    SELECT
        stay_id,
        subject_id,
        AVG(weight) AS weight -- Taking average weight during the stay
    FROM ( 
        WITH wt_stg AS (
            SELECT
                c.stay_id,
                c.subject_id,
                c.charttime,
                CASE WHEN c.itemid = 226512 THEN
                    'admit'
                ELSE
                    'daily'
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
-- Demographics with age and gender
demographics AS (
    SELECT
        ad.subject_id,
        ad.hadm_id,
        FLOOR(EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year + pa.anchor_age) AS age,
        -- Convert gender to binary (M=1, F=0)
        CASE WHEN pa.gender = 'M' THEN
            1
        ELSE
            0
        END AS gender_binary
    FROM
        mimiciv_hosp.admissions ad
        INNER JOIN mimiciv_hosp.patients pa ON ad.subject_id = pa.subject_id
    WHERE
        FLOOR(EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year + pa.anchor_age) BETWEEN 16 AND 89
),
-- ICU readmission flag
icu_readmit AS (
    SELECT
        stay_id,
        subject_id,
        hadm_id,
        CASE WHEN ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) > 1 THEN
            1
        ELSE
            0
        END AS icu_readmit
    FROM
        mimiciv_icu.icustays
),
-- Charlson Comorbidity Index
cci AS (
    SELECT
        subject_id,
        hadm_id,
        charlson_comorbidity_index
    FROM
        mimiciv_derived.charlson
),
-- Combine all measurements with AKI patients
combined_data AS (
    SELECT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        d.age,
        d.gender_binary AS gender, -- Now using binary gender
        h.height,
        w.weight,
        CASE WHEN h.height IS NOT NULL
            AND w.weight IS NOT NULL THEN
            ROUND(CAST((w.weight / POWER(h.height / 100, 2)) AS numeric), 2)
        ELSE
            NULL
        END AS bmi,
        r.icu_readmit,
        c.charlson_comorbidity_index,
        -- Add row number for deduplication
        ROW_NUMBER() OVER (PARTITION BY f.subject_id ORDER BY f.hadm_id,
            f.stay_id) AS rn
    FROM
        final_aki_status f
        LEFT JOIN demographics d ON f.subject_id = d.subject_id
            AND f.hadm_id = d.hadm_id
        LEFT JOIN height_merged h ON f.subject_id = h.subject_id
            AND f.stay_id = h.stay_id
        LEFT JOIN weight_data w ON f.subject_id = w.subject_id
            AND f.stay_id = w.stay_id
        LEFT JOIN icu_readmit r ON f.subject_id = r.subject_id
            AND f.stay_id = r.stay_id
        LEFT JOIN cci c ON f.subject_id = c.subject_id
            AND f.hadm_id = c.hadm_id
)
-- Select only the first record for each subject_id
SELECT
    subject_id,
    age,
    gender, -- 1=Male, 0=Female
    height,
    weight,
    bmi,
    icu_readmit,
    charlson_comorbidity_index
FROM
    combined_data
WHERE
    rn = 1
ORDER BY
    subject_id;

-- Export demographics to shared directory
\COPY (SELECT * FROM temp_aki_demographics) TO '/home/hwxu/Projects/Research/PKU/AAAI2026/Input/raw/shared/aki_demographics.csv' WITH CSV HEADER DELIMITER ','; 