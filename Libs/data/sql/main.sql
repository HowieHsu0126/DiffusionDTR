-- ========================================================================
-- MAIN SQL SCRIPT FOR ICU-AKI RESEARCH PROJECT (AAAI 2026)
-- PoG (Plan-on-Graph) Model Data Preparation
--
-- This script generates all necessary CSV files for:
-- 1. Mechanical Ventilation (vent) strategy optimization
-- 2. Renal Replacement Therapy (rrt) strategy optimization
--
-- Output Structure:
-- - Input/raw/shared/: Shared data (demographics, comorbidity, labs, vitals, SOFA)
-- - Input/raw/task/: Task-specific data (ventilation, RRT, IV fluids/vasopressors)
-- ========================================================================
-- ========================================================================
-- TIME WINDOW CONFIGURATION (EASILY MODIFIABLE)
-- ========================================================================
-- Configure the time window around AKI diagnosis for data extraction
-- Modify these values to change the observation period
\set HOURS_BEFORE_AKI 24
\set HOURS_AFTER_AKI 72
-- Alternative configurations (uncomment to use):
-- \set HOURS_BEFORE_AKI 0    -- Start from AKI diagnosis
-- \set HOURS_AFTER_AKI 72    -- 72 hours after AKI diagnosis
-- \set HOURS_BEFORE_AKI 12   -- 12 hours before AKI diagnosis
-- \set HOURS_AFTER_AKI 96    -- 96 hours after AKI diagnosis
\echo 'Time window configuration:'
\echo '  Before AKI diagnosis: ' :HOURS_BEFORE_AKI ' hours'
\echo '  After AKI diagnosis: ' :HOURS_AFTER_AKI ' hours'
\echo '  Total observation window: ' (:HOURS_BEFORE_AKI + :HOURS_AFTER_AKI) ' hours'
\echo ''
-- Ensure output directories exist
\! mkdir -p '/home/hwxu/Projects/Research/PKU/AAAI2026/Input/raw/shared'
\! mkdir -p '/home/hwxu/Projects/Research/PKU/AAAI2026/Input/raw/task'
\echo 'Starting ICU-AKI data extraction...'
-- ========================================================================
-- SECTION 1: SHARED DATA (exported to shared/)
-- ========================================================================
-- 1. Demographics
\i '/home/hwxu/Projects/Research/PKU/AAAI2026/Libs/data/sql/shared/demographics.sql'
-- 2. Comorbidity
\i '/home/hwxu/Projects/Research/PKU/AAAI2026/Libs/data/sql/shared/comorbidity.sql'
-- 3. SOFA Score Time Series
\i '/home/hwxu/Projects/Research/PKU/AAAI2026/Libs/data/sql/shared/sofa.sql'
-- 4. Laboratory Results Time Series
\i '/home/hwxu/Projects/Research/PKU/AAAI2026/Libs/data/sql/shared/laboratory.sql'
-- 5. Vital Signs Time Series
\i '/home/hwxu/Projects/Research/PKU/AAAI2026/Libs/data/sql/shared/vitalsigns.sql'
-- 6. Mortality Outcomes (90-day mortality)
\i '/home/hwxu/Projects/Research/PKU/AAAI2026/Libs/data/sql/shared/mortality.sql'
-- ========================================================================
-- SECTION 2: TASK-SPECIFIC DATA (exported to task/)
-- ========================================================================
\echo 'Generating task-specific data...'
-- 6. IV Fluids and Vasopressor Analysis (5-level binning for DTR optimization)
\echo 'Generating IV fluids and vasopressor data...'
\i '/home/hwxu/Projects/Research/PKU/AAAI2026/Libs/data/sql/task/ivfluids_vasopressor.sql'
-- 7. Mechanical Ventilation Time Series
\echo 'Generating mechanical ventilation data...'
\i '/home/hwxu/Projects/Research/PKU/AAAI2026/Libs/data/sql/task/ventilation.sql'
-- 8. RRT Time Series with Combinational Action Space
\echo 'Generating renal replacement therapy data...'
\i '/home/hwxu/Projects/Research/PKU/AAAI2026/Libs/data/sql/task/rrt.sql'
-- ========================================================================
-- CLEANUP
-- ========================================================================
\echo 'Cleaning up temporary tables...'
-- Clean up all temporary tables
DROP TABLE IF EXISTS temp_aki_demographics;

DROP TABLE IF EXISTS temp_aki_comorbidity;

DROP TABLE IF EXISTS temp_aki_sofa_timeseries;

DROP TABLE IF EXISTS temp_aki_labres_timeseries;

DROP TABLE IF EXISTS temp_aki_vitalsigns_gcs_timeseries;

DROP TABLE IF EXISTS temp_aki_90day_mortality;

DROP TABLE IF EXISTS temp_aki_ivfluids_vasopressor;

DROP TABLE IF EXISTS temp_aki_vent_timeseries;

DROP TABLE IF EXISTS temp_aki_rrt_timeseries;

DROP TABLE IF EXISTS vitalsign;

\echo 'ICU-AKI data extraction completed successfully!'
\echo ''
\echo 'Generated files:'
\echo '  Input/raw/shared/:'
\echo '    - aki_demographics.csv (demographic data)'
\echo '    - aki_comorbidity.csv (comorbidity data)'
\echo '    - sofa_aki.csv (SOFA scores time series)'
\echo '    - aki_labres_timeseries.csv (laboratory results time series)'
\echo '    - aki_vitalsigns_timeseries.csv (vital signs time series)'
\echo '    - aki_90day_mortality.csv (90-day mortality outcomes)'
\echo ''
\echo '  Input/raw/task/:'
\echo '    - aki_ivfluids_vasopressor.csv (IV fluids and vasopressor data)'
\echo '    - aki_vent.csv (mechanical ventilation data)'
\echo '    - aki_rrt.csv (renal replacement therapy data)'
\echo ''
\echo 'Raw data has been extracted from MIMIC-IV successfully!' 
