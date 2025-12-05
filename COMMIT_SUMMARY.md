# Commit Summary

## Permutation Importance Updates

### 1. Metric Definition Change
- Switched from **Delta Accuracy** (Permuted - Baseline) to **Decrease in Accuracy** (Baseline - Permuted).
- **Interpretation:** Positive values now indicate **importance** (feature contributes to accuracy). Negative values indicate the feature was harmful or noise.
- This aligns with standard interpretation: "Larger positive value = More important".

### 2. Visualization Updates
- **Sorting:** Plots are now sorted in **descending order** of importance.
- **Top:** Most predictive features.
- **Cutoff:** `random_noise` (red bar) serves as a visual cutoff. Features below it are non-informative.
- **Labels:** Updated x-axis label to "Decrease in Accuracy (Baseline - Permuted)".

### 3. Paper Text & Captions
- **Captions:** Explicitly explain the ranking logic and the role of `random_noise` as a validity cutoff.
- **Results Section:** Rewrote the description to focus on the hierarchy: "High-importance features at the top... random noise anchoring the baseline."
- **Clarification:** Removed confusing references to "negative values" being important.

## Files Modified
- `replicate_analysis.py`: Updated calculation and sorting logic.
- `tornado_vulnerability_paper_updated.tex`: Updated text and captions.
- `tornado_vulnerability_outputs/`: Regenerated all plots and CSVs.
