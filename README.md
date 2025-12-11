# Feature Importance Analysis for Tornado Vulnerability

This repository contains the code and data for the paper "Identifying Vulnerability Factors in Historic Buildings under Tornado Loading". The study employs a multi-model machine learning framework with permutation importance and SHAP analysis to generate testable hypotheses for preservation engineering.

## Data Files

The analysis uses two updated datasets located in the `updatedData/` directory:
- `Nashville_Tornado_DataInput_Final_111425(in).csv`: 2020 Nashville EF3-EF4 tornado
- `Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv`: 2021 Quad State EF4 tornado (Mayfield, KY)

## Key Scripts

### 1. Main Analysis Pipeline
*   `replicate_analysis.py`: **Start here.** Runs the full ML pipeline (Random Forest, XGBoost, etc.), performs permutation importance, and generates performance metrics.
    *   **Outputs:** `model_performance_cv.csv`, `permutation_importance.csv`, `statistical_equivalence.csv`

### 2. Mechanistic Interpretation
*   `shap_analysis.py`: Performs detailed SHAP analysis to explain *how* the models work. Generates beeswarm plots and dependence plots for all damage classes (Undamaged, Low, Significant).
    *   **Outputs:** `shap_beeswarm_class*.png`, `shap_model_comparison.csv`

### 3. Data Visualization & Statistics
*   `generate_supp_plots.py`: Generates supplementary figures (Age distribution, Damage distribution) and performs statistical tests (Mann-Whitney U).
*   `get_paper_stats.py`: Utility script to print out specific numerical values cited in the paper text (e.g., correlations, F1 scores).
*   `generate_shap_dependence.py`: Specialized script for generating SHAP interaction plots with threshold lines.

## Prerequisites

Python 3.8+ with the following packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap imbalanced-learn scipy openpyxl
```

## Running the Analysis

```bash
# 1. Run the main model training and permutation importance
python3 replicate_analysis.py

# 2. Run SHAP analysis for interpretation
python3 shap_analysis.py

# 3. Generate supplementary plots
python3 generate_supp_plots.py
```

## Key Methodology Features

### Data Preprocessing
- **Unified Schema:** Standardized handling of "unknown" values across Nashville and Quad State datasets.
- **Handling Missing Data:** Median imputation for numeric features; informative "unknown/missing" category preserved for categorical features.
- **Class Imbalance:** Handled using SMOTENC oversampling within cross-validation folds.

### Models Benchmarked
- **Ensemble:** Random Forest, XGBoost
- **Linear/Simple:** Linear SVC, Logistic Regression, Ridge Classifier, Decision Tree

All models are evaluated using repeated stratified 5-fold cross-validation (25 total folds) with statistical equivalence testing (Wilcoxon signed-rank test with Holm-Bonferroni correction).

### Feature Comparison
- **Permutation Importance:** Used as a "Gatekeeper" to identify features that globally outperform a random noise baseline.
- **SHAP Analysis:** Used as an "Explainer" to identify instance-level drivers and interaction effects for validated models.

## Output Directory

All outputs (CSV tables, PNG figures) are saved to `tornado_vulnerability_outputs/`. The LaTeX manuscript references figures directly from this directory.

## License

Data available via DesignSafe-CI Project PRJ-3417: https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3417
