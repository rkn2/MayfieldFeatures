# Paper Consistency Review - Changes Made

## Issue Identified
The paper incorrectly stated that "Roof System" was a strong predictor, when in fact it ranked **last** (32nd out of 32 features) with a **negative** importance score (-0.006).

## Actual Top Predictors (Hazard-Neutral Setting)
Based on the permutation importance analysis:

1. **retrofit_type_u** (0.022) - Strongest
2. **roof_slope_u** (0.022) - Second strongest
3. **structural_wall_system_u** (0.021)
4. **mwfrs_u_wall** (0.017)
5. **wall_cladding_u** (0.012)
6. **occupany_u** (0.011)
7. **retrofit_present_u** (0.010)
8. **buidling_height_m** (0.009)

**Bottom predictors:**
- **roof_shape_u** (rank 31, -0.002)
- **roof_system_u** (rank 32, -0.006) - LAST

## Changes Made

### 1. Abstract (Line 54)
**Before:** "roof system characteristics and wall-to-diaphragm connections"
**After:** "retrofit features, structural wall systems, and roof slope"

### 2. Results Section (Line 400)
**Before:** "high-importance structural features (roof system, MWFRS)"
**After:** "high-importance features (retrofit type, roof slope, structural wall system, MWFRS)"

### 3. Physical Interpretation Section (Lines 408-414)
**Before:** Discussed "Roof System" and "MWFRS" as strong predictors
**After:** Discusses "retrofit features, roof slope, and structural wall systems" as strong predictors

**Key changes:**
- Removed references to `roof_system_u` as a predictor
- Added discussion of retrofit features (which are actually top predictors)
- Clarified that roof **slope** (not shape or system) is the important roof-related feature
- Updated the figure caption to note that roof shape is descriptive data, not a model finding

### 4. Discussion Section (Line 553)
**Before:** "Roof System Vulnerabilities" subsection title
**After:** "Roof and Structural System Vulnerabilities"

**Before:** "Roof systems consistently ranked as the primary predictor"
**After:** "Retrofit features and roof slope consistently ranked among the primary predictors"

### 5. Conclusions (Line 644)
**Before:** "roof system characteristics and wall-to-diaphragm connections"
**After:** "retrofit features, roof slope, structural wall systems, and wall-to-diaphragm connections"

### 6. Figure Caption (Line 419)
**Before:** "supporting the model findings"
**After:** "While roof shape itself did not emerge as a strong predictor in the permutation importance analysis, this descriptive plot shows that hip roofs had a higher proportion of undamaged buildings compared to gable roofs in the raw data."

## Verification
All changes have been verified against the actual permutation importance results in:
`tornado_vulnerability_outputs/permutation_importance.csv`

The paper now accurately reflects which features are actually important predictors versus which are not.
