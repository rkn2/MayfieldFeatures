# Tornado Vulnerability Feature Importance Analysis

This repository contains the code and analysis for examining the relationship between historic building features and tornado damage using machine learning and statistical validation.

## Overview

This project analyzes building-level damage data from the 2020 Nashville and 2021 Quad State tornadoes to identify potential vulnerability factors in historic unreinforced masonry (URM) structures. The analysis uses a multi-model machine learning framework with permutation importance and SHAP analysis to generate testable hypotheses for preservation engineering.

## Prerequisites

Python 3.8+ with the following packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap imbalanced-learn scipy openpyxl
```

## Data Files

The analysis uses two datasets:
- `Nashville_Tornado_DataInput_Final_110725.xlsx`: 2020 Nashville EF3-EF4 tornado
- `QuadState_Tornado_DataInputv2.csv`: 2021 Quad State EF4 tornado (Mayfield, KY)

## Running the Analysis

### 1. Main Analysis (Model Training and Permutation Importance)

```bash
python3 replicate_analysis.py
```

Outputs:
- `tornado_vulnerability_outputs/model_performance_cv.csv`: Cross-validation metrics for all models
- `tornado_vulnerability_outputs/statistical_equivalence.csv`: Wilcoxon test results
- `tornado_vulnerability_outputs/permutation_importance.csv`: Feature importance scores
- `tornado_vulnerability_outputs/delta_accuracy_*.png`: Permutation importance plots

### 2. SHAP Analysis (Mechanistic Interpretation)

```bash
python3 shap_analysis.py
```

Outputs:
- `tornado_vulnerability_outputs/shap_beeswarm_class*.png`: SHAP summary plots for each damage class
- `tornado_vulnerability_outputs/shap_dependence_*.png`: Feature interaction plots
- `tornado_vulnerability_outputs/shap_model_comparison.csv`: Cross-model SHAP validation

### 3. Supplementary Figures (Data Exploration)

```bash
python3 generate_supplementary_plots.py
```

Outputs:
- `tornado_vulnerability_outputs/supp_year_built_by_event.png`: Age distribution
- `tornado_vulnerability_outputs/supp_damage_by_event.png`: Damage class proportions
- `tornado_vulnerability_outputs/supp_year_built_by_damage.png`: Age vs. damage boxplot
- `tornado_vulnerability_outputs/supp_roof_shape_damage.png`: Roof geometry analysis
- `tornado_vulnerability_outputs/supp_ef_rating_dist.png`: EF rating distribution

## Key Features

### Data Preprocessing
- Standardized handling of "unknown" values across datasets
- Median imputation for numeric features
- Ordinal encoding for categorical features with unknown handling
- SMOTENC oversampling within cross-validation folds

### Feature Set
**Numeric Features**: Stories, year built, building area, height, wall dimensions, wall thickness, parapet height, overhang length, fenestration percentages (N/S/E/W)

**Categorical Features**: Archetype, occupancy, urban setting, roof shape/slope, construction type, MWFRS (wall/roof), structural wall system, foundation type, wall substrate/cladding, roof system/substrate/cover, retrofit presence/type

**Hazard Features**: EF rating (numeric), distance to tornado track (km)

### Models Benchmarked
- Random Forest
- XGBoost
- Decision Tree
- Linear SVC
- Logistic Regression
- Ridge Classifier

All models use balanced class weights and repeated stratified 5-fold cross-validation (5 repeats = 25 folds).

## Output Directory

All outputs are saved to `tornado_vulnerability_outputs/`. The LaTeX manuscript references figures directly from this directory.

## Notes

- The analysis includes a random noise feature as a negative control
- Statistical equivalence testing uses Wilcoxon signed-rank with Holm-Bonferroni correction
- SHAP analysis is computed on held-out validation data only (no synthetic SMOTENC samples)
- Missing data is treated as informative (categorical "unknown" category preserved)

## Citation

If you use this code or data, please cite:

Kaushal, S.S., Gutierrez Soto, M., & Napolitano, R. (2024). Examining the Relationships Between Historic Building Features and Tornado Damage: A Multi-Model Feature Importance Analysis with Statistical Validation. *Engineering Structures* (in preparation).

## License

Data available via DesignSafe-CI Project PRJ-3417: https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3417

