# Final Corrections - Permutation Importance Analysis

## What Was Wrong

I initially calculated feature importance using ALL 6 models, but the paper correctly only reports results from the 2 statistically equivalent models (RandomForest and XGBoost). This led to completely different rankings.

## Correct Top Predictors (RandomForest + XGBoost only)

1. **mwfrs_u_wall** (MWFRS for walls) - 0.013
2. **occupany_u** (Occupancy type) - 0.012
3. **parapet_height_m** (Parapet height) - 0.008
4. **foundation_type_u** (Foundation type) - 0.004
5. **wall_thickness** - 0.003
6. **random_noise** - 0.002 ← CUTOFF

Everything below random noise is not a reliable predictor.

## Features That Are NOT Top Predictors

These ranked below random noise in the correct analysis:
- retrofit_type_u (rank 32, LAST)
- structural_wall_system_u (rank 31)
- roof_slope_u (rank 16)
- roof_system_u (rank 30)
- roof_shape_u (rank 17)

## Changes Made to Paper

### Abstract
- Changed from "retrofit features, structural wall systems, and roof slope"
- To: "MWFRS for walls, occupancy type, and parapet height"

### Results Section
- Updated description of top features to match actual rankings
- Added explanation that most features (including retrofit, roof slope) are below random noise
- Removed figure about roof shape (not a top predictor)

### Discussion Section
- Rewrote "Structural System and Geometric Vulnerabilities" to focus on MWFRS and parapet height
- Removed claims about roof geometries and hurricane straps (not supported by data)

### Conclusions
- Updated summary to reflect correct predictors

## Writing Style

All changes follow minimalist editing principles:
- Used contractions (don't, can't, we've)
- Removed AI buzzwords
- No listicles or bullet points in the paper text
- Simple, direct language
