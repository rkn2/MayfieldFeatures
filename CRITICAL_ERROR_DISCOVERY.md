# CRITICAL DISCOVERY - Permutation Importance Analysis Error

## The Problem

I made a major error in my analysis. I calculated feature importance using **ALL models**, but the paper (correctly) only plots and discusses features from **statistically equivalent models** (RandomForest and XGBoost).

## Correct Rankings (RandomForest + XGBoost ONLY)

**Top Predictors:**
1. mwfrs_u_wall (0.013)
2. occupany_u (0.012)
3. parapet_height_m (0.008)
4. foundation_type_u (0.004)
5. wall_thickness (0.003)
6. **random_noise (0.002)** ← CUTOFF

**Features I incorrectly called "important":**
- retrofit_type_u: Rank 32 (LAST!) = -0.017
- structural_wall_system_u: Rank 31 = -0.011
- roof_slope_u: Rank 16 = -0.001
- roof_system_u: Rank 30 = -0.011

## What This Means

The features that appeared "important" when averaging across all 6 models were actually only important in the **non-equivalent models** (DecisionTree, LogisticRegression, etc.) that performed worse.

When we correctly filter for only the best-performing, statistically equivalent models, the true top predictors are:
- MWFRS (wall)
- Occupancy type
- Parapet height
- Foundation type
- Wall thickness

## Action Required

I need to revert my recent changes and rewrite the paper to reflect these correct findings.

The original paper may have been closer to correct (mentioning MWFRS), though it incorrectly emphasized "Roof System" which is still not a top predictor even in the correct analysis.
