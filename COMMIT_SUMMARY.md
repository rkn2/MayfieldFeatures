# Commit Summary

## Major Updates

### Paper Revisions (tornado_vulnerability_paper_updated.tex)
1. **Missing Data Transparency**: Added missingness percentages to Table 1 and discussion of informative missingness
2. **Fenestration Mechanism**: Added detailed subsection with three competing hypotheses (dominant opening, structural weakening, confounding)
3. **Age-Damage Analysis**: Explained survivorship bias in pre-1880 buildings
4. **Reversibility Framework**: Distinguished mechanical vs. material reversibility in intervention table
5. **Risk-Informed Decision-Making**: Added new section with three-tier intervention framework
6. **Language Cleanup**: Removed AI buzzwords (robust, critically, underscores, comprehensive) and em-dashes

### Code Updates
1. **Data Cleaning**: Standardized "unknown" value handling across datasets (Nashville vs. QuadState)
2. **Feature Set**: Added fenestration features (wall_fenestration_per_n/s/e/w), removed first_floor_elevation_m
3. **SHAP Analysis**: Updated to match main analysis feature set
4. **Supplementary Plots**: Ensured consistency with data cleaning

### Documentation
1. **README.md**: Complete rewrite with clearer structure, feature descriptions, and citation info
2. **PAPER_UPDATES_SUMMARY.md**: Detailed changelog for all revisions

### Bibliography
1. Added Mehta (1976) and Marshall (1977) citations for wind engineering references

## Files Modified
- tornado_vulnerability_paper_updated.tex
- README.md
- replicate_analysis.py
- shap_analysis.py
- generate_supplementary_plots.py
- tornado_refs.bib

## Files Removed
- analyze_missingness.py (temporary analysis)
- check_correlations.py (temporary analysis)
- get_paper_metrics.py (temporary analysis)

## Ready to Commit
All temporary files removed. Core analysis scripts cleaned and documented.
