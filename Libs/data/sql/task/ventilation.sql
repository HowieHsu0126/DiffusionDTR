-- ========================================================================
-- MECHANICAL VENTILATION DATA EXTRACTION
-- Generates aki_vent.csv for vent/ directory only
-- Uses HOURS_BEFORE_AKI and HOURS_AFTER_AKI variables
-- ========================================================================

\echo 'Generating ventilation-specific data...'

-- 7. Mechanical Ventilation Time Series
DROP TABLE IF EXISTS temp_aki_vent_timeseries;

CREATE TEMP TABLE temp_aki_vent_timeseries AS
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
-- 获取患者身高和性别信息，用于计算理想体重
patient_info AS (
    SELECT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        COALESCE(ht_cm.height, ROUND(CAST((ht_in.height * 2.54) AS numeric), 2)) AS height_cm,
        CASE WHEN pa.gender = 'M' THEN
            1
        ELSE
            0
        END AS gender
    FROM
        final_aki_status f
        JOIN first_icu fi ON f.subject_id = fi.subject_id
            AND f.hadm_id = fi.hadm_id
            AND f.stay_id = fi.stay_id
        LEFT JOIN mimiciv_hosp.patients pa ON f.subject_id = pa.subject_id
        LEFT JOIN (
            SELECT
                subject_id,
                stay_id,
                AVG(valuenum) AS height
            FROM
                mimiciv_icu.chartevents
            WHERE
                itemid = 226730
                AND valuenum > 120
                AND valuenum < 230
            GROUP BY
                subject_id,
                stay_id) ht_cm ON f.subject_id = ht_cm.subject_id
            AND f.stay_id = ht_cm.stay_id
        LEFT JOIN (
            SELECT
                subject_id,
                stay_id,
                AVG(valuenum) AS height
            FROM
                mimiciv_icu.chartevents
            WHERE
                itemid = 226707
                AND valuenum > 47
                AND valuenum < 90
            GROUP BY
                subject_id,
                stay_id) ht_in ON f.subject_id = ht_in.subject_id
            AND f.stay_id = ht_in.stay_id
),
ibw_calc AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        height_cm,
        gender,
        CASE WHEN gender = 1 THEN
            50 + 0.91 * (height_cm - 152.4)
        ELSE
            45.5 + 0.91 * (height_cm - 152.4)
        END AS ideal_body_weight
    FROM
        patient_info
    WHERE
        height_cm IS NOT NULL
),
vent_events AS (
    SELECT
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        vs.charttime,
        f.earliest_aki_diagnosis_time,
        ROUND(EXTRACT(EPOCH FROM (vs.charttime - f.earliest_aki_diagnosis_time)) / 3600) AS hours_from_onset,
        NULL::integer AS rrt_present,
        NULL::integer AS rrt_active,
        NULL::text AS rrt_type,
        CASE WHEN od.o2_delivery_device_1 IN ('Tracheostomy tube') THEN
            'Trach'
        WHEN od.o2_delivery_device_1 IN ('Endotracheal tube')
            OR vs.ventilator_mode IN ('(S) CMV', 'APRV', 'APRV/Biphasic+ApnPress', 'APRV/Biphasic+ApnVol', 'APV (cmv)', 'Ambient', 'Apnea Ventilation', 'CMV', 'CMV/ASSIST', 'CMV/ASSIST/AutoFlow', 'CMV/AutoFlow', 'CPAP/PPS', 'CPAP/PSV+Apn TCPL', 'CPAP/PSV+ApnPres', 'CPAP/PSV+ApnVol', 'MMV', 'MMV/AutoFlow', 'MMV/PSV', 'MMV/PSV/AutoFlow', 'P-CMV', 'PCV+', 'PCV+/PSV', 'PCV+Assist', 'PRES/AC', 'PRVC/AC', 'PRVC/SIMV', 'PSV/SBT', 'SIMV', 'SIMV/AutoFlow', 'SIMV/PRES', 'SIMV/PSV', 'SIMV/PSV/AutoFlow', 'SIMV/VOL', 'SYNCHRON MASTER', 'SYNCHRON SLAVE', 'VOL/AC')
            OR vs.ventilator_mode_hamilton IN ('APRV', 'APV (cmv)', 'Ambient', '(S) CMV', 'P-CMV', 'SIMV', 'APV (simv)', 'P-SIMV', 'VS', 'ASV') THEN
            'InvasiveVent'
        WHEN od.o2_delivery_device_1 IN ('Bipap mask ', 'CPAP mask ')
            OR vs.ventilator_mode_hamilton IN ('DuoPaP', 'NIV', 'NIV-ST') THEN
            'NonInvasiveVent'
        WHEN od.o2_delivery_device_1 IN ('High flow neb', 'High flow nasal cannula') THEN
            'HighFlow'
        WHEN od.o2_delivery_device_1 IN ('Nasal cannula', 'Face tent', 'Aerosol-cool', 'Non-rebreather', 'Venti mask ', 'Medium conc mask ', 'T-piece', 'Ultrasonic neb', 'Vapomist', 'Oxymizer') THEN
            'Oxygen'
        ELSE
            NULL
        END AS vent_type,
        vs.tidal_volume_set,
        CASE WHEN vs.tidal_volume_set IS NOT NULL
            AND ibw.ideal_body_weight IS NOT NULL THEN
            ROUND(CAST(vs.tidal_volume_set / ibw.ideal_body_weight AS numeric), 2)
        ELSE
            NULL
        END AS tidal_volume_ibw,
        vs.peep,
        CASE WHEN vs.fio2 > 1 THEN
            ROUND(CAST(vs.fio2 / 100.0 AS numeric), 3)
        ELSE
            ROUND(CAST(vs.fio2 AS numeric), 3)
        END AS fio2
    FROM
        final_aki_status f
        JOIN first_icu fi ON f.subject_id = fi.subject_id
            AND f.hadm_id = fi.hadm_id
            AND f.stay_id = fi.stay_id
        LEFT JOIN mimiciv_derived.ventilator_setting vs ON f.stay_id = vs.stay_id
        LEFT JOIN mimiciv_derived.oxygen_delivery od ON f.stay_id = od.stay_id
            AND vs.charttime = od.charttime
        LEFT JOIN ibw_calc ibw ON f.stay_id = ibw.stay_id
    WHERE
        f.earliest_aki_diagnosis_time IS NOT NULL
                AND vs.charttime BETWEEN f.earliest_aki_diagnosis_time - (:HOURS_BEFORE_AKI || ' hours')::INTERVAL
            AND f.earliest_aki_diagnosis_time + (:HOURS_AFTER_AKI || ' hours')::INTERVAL
),
vent_events_dedup AS (
    SELECT DISTINCT ON (subject_id,
        hadm_id,
        stay_id,
        charttime)
        subject_id,
        hadm_id,
        stay_id,
        charttime,
        earliest_aki_diagnosis_time,
        hours_from_onset,
        rrt_present,
        rrt_active,
        rrt_type,
        vent_type,
        tidal_volume_set,
        tidal_volume_ibw,
        peep,
        fio2
    FROM
        vent_events
    ORDER BY
        subject_id,
        hadm_id,
        stay_id,
        charttime,
        vent_type DESC
),
ranked AS (
    SELECT
        subject_id,
        hours_from_onset,
        -- 离散化 peep
        CASE WHEN peep IS NULL THEN
            NULL
        WHEN peep <= 5 THEN
            0
        WHEN peep <= 7 THEN
            1
        WHEN peep <= 9 THEN
            2
        WHEN peep <= 11 THEN
            3
        WHEN peep <= 13 THEN
            4
        WHEN peep <= 15 THEN
            5
        ELSE
            6
        END AS peep_bin,
        -- 离散化 fio2（假设单位为百分比，如果为小数需乘以100）
        CASE WHEN fio2 IS NULL THEN
            NULL
        WHEN fio2 * 100 <= 30 THEN
            0
        WHEN fio2 * 100 <= 35 THEN
            1
        WHEN fio2 * 100 <= 40 THEN
            2
        WHEN fio2 * 100 <= 45 THEN
            3
        WHEN fio2 * 100 <= 50 THEN
            4
        WHEN fio2 * 100 <= 55 THEN
            5
        ELSE
            6
        END AS fio2_bin,
        -- 离散化 tidal_volume_ibw
        CASE WHEN tidal_volume_ibw IS NULL THEN
            NULL
        WHEN tidal_volume_ibw <= 0 THEN
            0
        WHEN tidal_volume_ibw <= 5 THEN
            1
        WHEN tidal_volume_ibw <= 7.5 THEN
            2
        WHEN tidal_volume_ibw <= 10 THEN
            3
        WHEN tidal_volume_ibw <= 12.5 THEN
            4
        WHEN tidal_volume_ibw <= 15 THEN
            5
        ELSE
            6
        END AS tidal_volume_ibw_bin,
        ROW_NUMBER() OVER (PARTITION BY subject_id,
            hours_from_onset ORDER BY 
            (CASE WHEN peep IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN fio2 IS NOT NULL THEN
                0
            ELSE
                1
            END) + (CASE WHEN tidal_volume_ibw IS NOT NULL THEN
                0
            ELSE
                1
            END)) AS rn
    FROM
        vent_events_dedup
    WHERE
        hours_from_onset BETWEEN -:HOURS_BEFORE_AKI AND :HOURS_AFTER_AKI
)
SELECT
    subject_id,
    hours_from_onset,
    peep_bin,
    fio2_bin,
    tidal_volume_ibw_bin
FROM
    ranked
WHERE
    rn = 1
    AND peep_bin IS NOT NULL
    AND fio2_bin IS NOT NULL
    AND tidal_volume_ibw_bin IS NOT NULL
ORDER BY
    subject_id,
    hours_from_onset;

-- Export ventilation data to task directory
\COPY (SELECT * FROM temp_aki_vent_timeseries) TO '/home/hwxu/Projects/Research/PKU/AAAI2026/Input/raw/task/aki_vent.csv' WITH CSV HEADER DELIMITER ','; 