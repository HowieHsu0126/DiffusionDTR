# SQL Scripts Modular Structure

## Overview
The main SQL script has been modularized into separate files for better maintainability and organization. All scripts use the configurable time window variables `HOURS_BEFORE_AKI` and `HOURS_AFTER_AKI`.

## File Structure
```
Libs/data/sql/
├── main.sql                    # Main control script
├── shared/                     # Shared data scripts (exported to both vent/ and rrt/)
│   ├── demographics.sql        # Patient demographics
│   ├── comorbidity.sql         # Comorbidity data (Charlson Index)
│   ├── sofa.sql               # SOFA score time series
│   ├── laboratory.sql         # Laboratory results time series
│   ├── vitalsigns.sql         # Vital signs time series
│   └── mortality.sql          # 90-day mortality outcomes
├── treatment/
│   └── ivfluids_vasopressor.sql  # IV fluids and vasopressor data
├── vent/
│   └── ventilation.sql        # Mechanical ventilation data
└── rrt/
    └── rrt.sql       # Enhanced RRT data
```

## Usage

### Run All Scripts
```bash
psql -d your_database -f Libs/data/sql/main.sql
```

### Run Individual Scripts
You can run individual scripts if needed:
```bash
# Time window configuration must be set first
psql -d your_database -v HOURS_BEFORE_AKI=24 -v HOURS_AFTER_AKI=72 -f Libs/data/sql/shared/demographics.sql
```

## Configuration

### Time Window Settings
In `main.sql`, modify these variables to change the observation period:
```sql
\set HOURS_BEFORE_AKI 24    -- Hours before AKI diagnosis
\set HOURS_AFTER_AKI 72     -- Hours after AKI diagnosis
```

### Alternative Configurations
```sql
-- Start from AKI diagnosis
\set HOURS_BEFORE_AKI 0
\set HOURS_AFTER_AKI 72

-- Extended observation window
\set HOURS_BEFORE_AKI 12
\set HOURS_AFTER_AKI 96
```

## Generated Files

### Shared Data (in both vent/ and rrt/ directories)
- `aki_demographics.csv` - Patient demographics (age, gender, height, weight, BMI, etc.)
- `aki_comorbidity.csv` - Comorbidity data based on Charlson Comorbidity Index
- `sofa_aki.csv` - SOFA score time series
- `aki_labres_timeseries.csv` - Laboratory results time series
- `aki_vitalsigns_timeseries.csv` - Vital signs time series
- `aki_90day_mortality.csv` - 90-day mortality outcomes (binary survival labels)
- `aki_ivfluids_vasopressor.csv` - IV fluids and vasopressor data

### Ventilation-Specific Data (vent/ directory only)
- `aki_vent.csv` - Mechanical ventilation parameters (PEEP, FiO2, tidal volume)

### RRT-Specific Data (rrt/ directory only)
- `aki_rrt.csv` - Enhanced RRT parameters with combinational action space

## Key Features

1. **Modular Design**: Each data type is in a separate script for easy maintenance
2. **Configurable Time Windows**: Use variables for consistent time window configuration
3. **Shared Data**: Common data is exported to both directories automatically
4. **Specialized Data**: Treatment-specific data is only exported to relevant directories
5. **Clean Structure**: Main script acts as orchestrator, individual scripts handle specific data extraction

## Dependencies
- All scripts depend on the `final_aki_status` table being available
- Laboratory script creates a `vitalsign` table that is used by vital signs script
- Patient weight and height calculations are included where needed for dosing calculations

## Maintenance
- To add new shared data: Create a new script in `shared/` and add it to `main.sql`
- To add ventilation-specific data: Modify `vent/ventilation.sql`
- To add RRT-specific data: Modify `rrt/rrt.sql`
- To change time windows: Only modify the variables in `main.sql` 