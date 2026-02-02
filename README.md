# Feature Importance Analysis for Tornado Vulnerability

This repository contains the code and data for the paper "Identifying Vulnerability Factors in Historic Buildings under Tornado Loading". The study employs a multi-model machine learning framework with permutation importance and SHAP analysis to generate testable hypotheses for preservation engineering.

## Data Files

The analysis uses two updated datasets located in the `updatedData/` directory:
- `Nashville_Tornado_DataInput_Final_111425(in).csv`: 2020 Nashville EF3-EF4 tornado
- `Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv`: 2021 Quad State EF4 tornado (Mayfield, KY)

## Key Methodology Update (Distance-Based Model)

This repository has been updated to focus on a **Distance-Based Model** for tornado vulnerability. 
*   **Rationale:** To avoid circularity, **EF Ratings** are excluded from the feature set, as they are often post-hoc derived from the damage itself.
*   **Exposure Proxy:** **Distance to the tornado path** is used as the primary proxy for hazard intensity. This allows the model to control for exposure (proximity to the vortex) while identifying intrinsic building features (e.g., roof properties, wall substrate) that modify performance.
*   **Goal:** The analysis seeks to identify *mechanistic* insights: given a certain exposure level (distance), what features make a building more or less likely to suffer significant damage?

### Updated Scripts
*   `replicate_analysis_damage_target.py`: **Main Script.** Runs the Distance-Based machine learning pipeline.
    *   **Outputs:** Saved to `tornado_vulnerability_outputs_damage_target/`.
*   `shap_analysis_distance.py`: Performs SHAP analysis specifically for the distance-based model.
*   `get_interactions.py`: Calculates and exports SHAP interaction values.

### 1. Main Analysis Pipeline
*   `replicate_analysis_damage_target.py`: Runs the RF training with Distance included.
    *   **Outputs:** `model_performance_cv.csv`, `permutation_importance.csv`, `statistical_equivalence.csv` in `tornado_vulnerability_outputs_damage_target/`

### 2. Mechanistic Interpretation
*   `shap_analysis_distance.py`: Generates the bee-swarm plots referenced in `main_distance.tex`.
    *   **Outputs:** `shap_beeswarm_class*.png`

### 3. Data Visualization & Statistics
*   `generate_dist_plot.py`: Generates the Distance vs. Damage distribution plot.
*   `generate_supp_plots.py`: Generates supplementary figures.

## Prerequisites

Python 3.8+ with the following packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap imbalanced-learn scipy openpyxl latex
```

## Running the Analysis

```bash
# 1. Run the distance-based model training and global importance
python3 replicate_analysis_damage_target.py

# 2. Run SHAP analysis for mechanistic interpretation
python3 shap_analysis_distance.py

# 3. Compute SHAP interactions
python3 get_interactions.py
```

## Output Directory

All outputs (CSV tables, PNG figures) for the current paper version are saved to `tornado_vulnerability_outputs_damage_target/`. The LaTeX manuscript `main_distance.tex` references figures from this directory.

## License

Data available via DesignSafe-CI Project PRJ-3417: https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3417
