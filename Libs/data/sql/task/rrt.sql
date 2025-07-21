-- ========================================================================
-- OPTIMIZED RRT DATA EXTRACTION - PERFORMANCE ENHANCED
-- Generates aki_rrt.csv for task/ directory only
-- Uses HOURS_BEFORE_AKI and HOURS_AFTER_AKI variables
-- ========================================================================

\echo 'Generating RRT-specific data...'

-- 8. Optimized RRT Time Series with Core Action Space
DROP TABLE IF EXISTS temp_aki_rrt_timeseries;

CREATE TEMP TABLE temp_aki_rrt_timeseries AS
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
-- 简化的患者体重计算 - 只使用kg，减少JOIN
patient_weight AS (
    SELECT
        f.subject_id,
        f.stay_id,
        AVG(ce.valuenum) AS weight_kg
    FROM
        final_aki_status f
        JOIN first_icu fi ON f.subject_id = fi.subject_id
            AND f.hadm_id = fi.hadm_id
            AND f.stay_id = fi.stay_id
        LEFT JOIN mimiciv_icu.chartevents ce ON f.stay_id = ce.stay_id
            AND ce.itemid = 224639 -- Daily Weight (kg) only
            AND ce.valuenum BETWEEN 30 AND 300
    GROUP BY
        f.subject_id,
        f.stay_id
),
-- RRT基础事件 - 包含类型识别
rrt_base_events AS (
    SELECT 
        f.subject_id,
        f.hadm_id,
        f.stay_id,
        pe.starttime AS charttime,
        f.earliest_aki_diagnosis_time,
        ROUND(EXTRACT(EPOCH FROM (pe.starttime - f.earliest_aki_diagnosis_time)) / 3600) AS hours_from_onset,
        -- 简化的RRT类型分类
        CASE WHEN pe.itemid IN (225802, 225803, 225809) THEN 'CRRT' -- 连续性肾脏替代治疗
             WHEN pe.itemid = 225441 THEN 'IHD' -- 间歇性血液透析
             WHEN pe.itemid = 225955 THEN 'SCUF' -- 缓慢连续超滤
             ELSE 'Other'
        END AS rrt_type
    FROM final_aki_status f
    JOIN first_icu fi ON f.subject_id = fi.subject_id
        AND f.hadm_id = fi.hadm_id
        AND f.stay_id = fi.stay_id
    INNER JOIN mimiciv_icu.procedureevents pe ON f.stay_id = pe.stay_id
        AND pe.itemid IN (225802, 225803, 225809, 225955, 225441, 225805)
    WHERE f.earliest_aki_diagnosis_time IS NOT NULL
        AND pe.starttime BETWEEN f.earliest_aki_diagnosis_time - (:HOURS_BEFORE_AKI || ' hours')::INTERVAL
            AND f.earliest_aki_diagnosis_time + (:HOURS_AFTER_AKI || ' hours')::INTERVAL
),
-- 简化的RRT参数 - 只保留核心参数，减少JOIN数量
rrt_parameters AS (
    SELECT
        rbe.subject_id,
        rbe.hadm_id,
        rbe.stay_id,
        rbe.charttime,
        rbe.hours_from_onset,
        rbe.rrt_type,
        -- 只保留最重要的血流速度参数
        ce_blood.valuenum AS blood_flow_rate,
        -- 只保留主要的透析液流速
        ce_dialysate.valuenum AS dialysate_flow_rate,
        -- 简化的RRT剂量计算 - 使用目标值或估算
        COALESCE(
            ce_goal.valuenum,
            CASE WHEN ce_dialysate.valuenum IS NOT NULL AND pw.weight_kg IS NOT NULL AND pw.weight_kg > 30 
                 THEN ce_dialysate.valuenum / pw.weight_kg
                 ELSE NULL END
        ) AS rrt_dose_ml_kg_hr
    FROM rrt_base_events rbe
    LEFT JOIN patient_weight pw ON rbe.subject_id = pw.subject_id
        AND rbe.stay_id = pw.stay_id
    -- 血流速度 - 保留最重要的参数
    LEFT JOIN mimiciv_icu.chartevents ce_blood ON rbe.stay_id = ce_blood.stay_id
        AND ce_blood.itemid = 224144 -- Blood Flow (ml/min)
        AND ce_blood.charttime BETWEEN rbe.charttime - INTERVAL '2 hour' AND rbe.charttime + INTERVAL '2 hour'
        AND ce_blood.valuenum BETWEEN 50 AND 500
    -- 透析液流速 - 保留主要参数
    LEFT JOIN mimiciv_icu.chartevents ce_dialysate ON rbe.stay_id = ce_dialysate.stay_id
        AND ce_dialysate.itemid = 224154 -- Dialysate Rate
        AND ce_dialysate.charttime BETWEEN rbe.charttime - INTERVAL '2 hour' AND rbe.charttime + INTERVAL '2 hour'
        AND ce_dialysate.valuenum BETWEEN 500 AND 5000
    -- 目标剂量 - 简化的剂量参考
    LEFT JOIN mimiciv_icu.chartevents ce_goal ON rbe.stay_id = ce_goal.stay_id
        AND ce_goal.itemid = 225183 -- Current Goal
        AND ce_goal.charttime BETWEEN rbe.charttime - INTERVAL '2 hour' AND rbe.charttime + INTERVAL '2 hour'
        AND ce_goal.valuenum BETWEEN 10 AND 100
),
-- 简化的抗凝检查 - 只检查最常用的肝素
rrt_with_anticoag AS (
    SELECT
        rp.*,
        -- 简化的抗凝标识 - 只检查肝素
        CASE WHEN EXISTS (
            SELECT 1 FROM mimiciv_icu.chartevents ce_hep 
            WHERE ce_hep.stay_id = rp.stay_id 
                AND ce_hep.itemid = 224145 -- Heparin Dose (per hour)
                AND ce_hep.charttime BETWEEN rp.charttime - INTERVAL '4 hour' AND rp.charttime + INTERVAL '1 hour'
                AND ce_hep.valuenum > 0
        ) THEN 1 ELSE 0 END AS anticoagulation_active
    FROM rrt_parameters rp
),
-- 去重并选择最佳记录
rrt_events_dedup AS (
    SELECT DISTINCT ON (subject_id, hadm_id, stay_id, hours_from_onset)
        subject_id,
        hadm_id,
        stay_id,
        hours_from_onset,
        rrt_type,
        blood_flow_rate,
        dialysate_flow_rate,
        rrt_dose_ml_kg_hr,
        anticoagulation_active
    FROM rrt_with_anticoag
    WHERE rrt_type IS NOT NULL
    ORDER BY
        subject_id,
        hadm_id,
        stay_id,
        hours_from_onset,
        -- 优先选择有更多参数的记录
        (CASE WHEN blood_flow_rate IS NOT NULL THEN 0 ELSE 1 END) + 
        (CASE WHEN dialysate_flow_rate IS NOT NULL THEN 0 ELSE 1 END) + 
        (CASE WHEN rrt_dose_ml_kg_hr IS NOT NULL THEN 0 ELSE 1 END)
),
-- 简化的离散化 - 减少复杂度
ranked AS (
    SELECT
        subject_id,
        hours_from_onset,
        -- RRT类型离散化 (简化为4个类别)
        CASE WHEN rrt_type = 'CRRT' THEN 1
             WHEN rrt_type = 'IHD' THEN 2
             WHEN rrt_type = 'SCUF' THEN 3
             ELSE 4 -- Other
        END AS rrt_type_bin,
        -- RRT剂量离散化 (简化为4个档次)
        CASE WHEN rrt_dose_ml_kg_hr IS NULL THEN 0
             WHEN rrt_dose_ml_kg_hr <= 20 THEN 1 -- 低剂量
             WHEN rrt_dose_ml_kg_hr <= 30 THEN 2 -- 标准剂量
             WHEN rrt_dose_ml_kg_hr <= 40 THEN 3 -- 高剂量
             ELSE 4 -- 超高剂量
        END AS rrt_dose_bin,
        -- 血流速度离散化 (简化为4个档次)
        CASE WHEN blood_flow_rate IS NULL THEN 0
             WHEN blood_flow_rate <= 120 THEN 1 -- 低流速
             WHEN blood_flow_rate <= 180 THEN 2 -- 中等流速
             WHEN blood_flow_rate <= 240 THEN 3 -- 高流速
             ELSE 4 -- 极高流速
        END AS blood_flow_bin,
        -- 抗凝状态 (保持二元)
        anticoagulation_active AS anticoagulation_bin,
        ROW_NUMBER() OVER (PARTITION BY subject_id, hours_from_onset 
                          ORDER BY (CASE WHEN rrt_type IS NOT NULL THEN 0 ELSE 1 END) + 
                                  (CASE WHEN rrt_dose_ml_kg_hr IS NOT NULL THEN 0 ELSE 1 END)) AS rn
    FROM rrt_events_dedup
    WHERE hours_from_onset BETWEEN -:HOURS_BEFORE_AKI AND :HOURS_AFTER_AKI
)
SELECT
    subject_id,
    hours_from_onset,
    rrt_type_bin, -- RRT模式 (1-4)
    rrt_dose_bin, -- RRT剂量 (0-4)
    blood_flow_bin, -- 血流速度 (0-4)
    anticoagulation_bin -- 抗凝状态 (0-1)
FROM ranked
WHERE rn = 1
    AND rrt_type_bin > 0 -- 只保留有RRT的时间点
ORDER BY subject_id, hours_from_onset;

-- Export optimized RRT data to task directory
\COPY (SELECT * FROM temp_aki_rrt_timeseries) TO '/home/hwxu/Projects/Research/PKU/AAAI2026/Input/raw/task/aki_rrt.csv' WITH CSV HEADER DELIMITER ','; 