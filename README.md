# Tornado Vulnerability Analysis: Replication Guide

This repository contains the code and data used to generate the analysis and figures for the paper on tornado vulnerability of historic unreinforced masonry buildings.

## 1. Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap imbalanced-learn
```

## 2. Data Files

The analysis relies on two primary datasets located in the root directory:
*   `Nashville_Tornado_DataInput_Final_110725.xlsx`: Data from the 2020 Nashville tornado.
*   `QuadState_Tornado_DataInputv2.csv`: Data from the 2021 Quad State (Mayfield) tornado.

## 3. Running the Analysis

### A. Main Feature Importance Analysis
To replicate the core machine learning analysis, permutation importance, and statistical tests:

```bash
python3 replicate_analysis.py
```
*   **Outputs**:
    *   `tornado_vulnerability_outputs/model_performance_cv.csv`: Cross-validation metrics.
    *   `tornado_vulnerability_outputs/permutation_importance.csv`: Feature importance scores.
    *   `tornado_vulnerability_outputs/delta_accuracy_*.png`: Permutation importance plots.

### B. SHAP Analysis (Mechanistic Interpretability)
To generate SHAP beeswarm and dependence plots:

```bash
python3 shap_analysis.py
```
*   **Outputs**:
    *   `tornado_vulnerability_outputs/shap_beeswarm_class*.png`: SHAP summary plots.
    *   `tornado_vulnerability_outputs/shap_dependence_*.png`: Interaction plots (e.g., EF vs. Elevation).

### C. Supplementary Figures (Data Distribution)
To generate the descriptive statistics plots (histograms, boxplots, etc.) used in the paper:

```bash
python3 generate_supplementary_plots.py
```
*   **Outputs**:
    *   `tornado_vulnerability_outputs/supp_*.png`: Figures showing age distribution, damage by event, roof shape analysis, etc.

## 4. Output Directory Structure

All generated figures and tables are saved to `tornado_vulnerability_outputs/`. The LaTeX manuscript is configured to read images directly from this folder.

## 5. Key Scripts Overview

| Script | Purpose |
| :--- | :--- |
| `replicate_analysis.py` | **Core Logic**. Loads data, preprocesses features, trains ML models (RF, XGB, SVM, etc.), and computes Permutation Importance. |
| `shap_analysis.py` | **Deep Dive**. Trains a dedicated Random Forest to calculate SHAP values for granular feature interpretation. |
| `generate_supplementary_plots.py` | **Visualization**. Generates standard data science plots (histograms, bars) with a consistent design system (Green/Yellow/Red). |

## 6. Troubleshooting

*   **Missing "Hip" Roofs**: If roof shape plots look wrong, ensure `generate_supplementary_plots.py` is filtering for the top 4 (not 3) categories.
*   **Gridlines**: The design system defaults to `sns.set_style("white")` to remove gridlines.
*   **Color Palette**: Damage colors are hardcoded: Green (Undamaged), Yellow (Low), Red (Significant).
