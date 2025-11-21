# Paper Updates Summary

## Changes Made to Address Reviewer Comments

### 1. Missing Data Reporting (Table 1)
**Location:** Section 2.2, Table 1 (Dataset Composition)

**Changes:**
- Added "Missing (%)" column to Table 1
- Reported missingness for all key features
- High missingness features identified:
  - Roof substrate: 53%
  - Retrofit type: 49%
  - Wall substrate: 47%
  - Wall thickness: 36%
  - Foundation type: 27%
  - Fenestration: 6-10%

**New Discussion Added:**
- Explained that missingness is likely **informative** rather than random
- Buildings with extensive damage had inaccessible interiors
- Remote-sensing assessments couldn't document hidden details
- For retrofit features, "missing" often means "no retrofit documented"
- Acknowledged that median imputation and categorical encoding allow model to learn from "unknown" status
- Recommended future work on multiple imputation

---

### 2. Fenestration Mechanism Discussion
**Location:** Section 4.3, new subsection "Fenestration and Internal Pressurization"

**Changes:**
- Added detailed discussion of **three competing hypotheses**:
  1. **Dominant Opening Effect**: Breached windows → internal pressurization → roof uplift
  2. **Structural Weakening**: High fenestration → reduced wall area → weakened envelope
  3. **Confounding by Building Use**: Fenestration proxies for commercial vs. residential typology

**New Citations Added:**
- Mehta et al. (1976) - Wind-induced pressures on buildings
- Marshall (1977) - Pressure distributions on rectangular buildings

**Key Points:**
- Explained dominant opening concept (internal pressure coefficient +0.2 → +0.8)
- Acknowledged SHAP cannot distinguish causal mechanisms
- Recommended future CFD modeling and fragility analysis

---

### 3. Age-Damage Analysis
**Location:** Section 2.3 (Target Variable and Class Distribution)

**New Analysis Added:**
- Pre-1880 buildings: 90.6% undamaged (LOWEST damage rate)
- 1900-1920 buildings: 64.6% undamaged (HIGHEST damage rate)

**Explanation Provided:**
- **Survivorship bias**: Pre-1880 buildings represent pre-selected sample of well-built structures
- Poorly constructed contemporaries were demolished decades ago
- 1900-1920 cohort includes rapid commercial development (lower quality)
- Pre-1880 buildings more likely to have historic designation → better maintenance
- Conclusion: Age alone is poor proxy without accounting for maintenance/construction quality

---

### 4. Retrofit Reversibility Clarification
**Location:** Section 5.2.3 (Preservation Philosophy and Reversibility)

**Major Conceptual Update:**
- Distinguished **mechanical reversibility** (can be removed) from **material reversibility** (restores original condition)

**Table 5 Updates:**
- Hurricane straps: Changed from "High compatibility" to "Moderate compatibility"
  - Mech: Yes / Mat: No
  - Lag bolts create permanent holes in timber
- All interventions now rated on both dimensions
- Added clarification that NPS considers this "minimally invasive" not "truly reversible"

**New Recommendation:**
- Future research should prioritize compression-based systems or friction connections that avoid penetrating fasteners

---

### 5. Risk-Informed Decision-Making (NEW SECTION)
**Location:** Section 5.3 (new subsection before Conclusions)

**Major Addition: Tiered Intervention Framework**

**Hazard Context:**
- EF4+ tornadoes: >1,000 year return period
- EF1-EF2 tornadoes: 50-100 year return period

**Three-Tier Approach:**

**Tier 1: Life-Safety Minimum (All Buildings)**
- Secure parapets, anchor chimneys, brace gable ends
- Addresses EF1-EF2 partial component failures
- Low-cost, high-frequency hazard mitigation

**Tier 2: Envelope Protection (Designated Properties)**
- Roof-to-wall connections, wall-to-diaphragm anchors
- Prevents total loss in EF2-EF3 events
- For buildings with formal historic designation

**Tier 3: Extreme Event Hardening (Exceptional Cases)**
- Foundation reinforcement, full diaphragm replacement
- Reserved for buildings of exceptional cultural significance
- Acknowledges even these may not survive EF5

**Limitations of Survival-Based Metrics:**
- **Interior Hazards**: Undamaged buildings may still have falling plaster, fixtures, partitions
- **Post-Event Functionality**: Low-damage buildings may lack utilities, weatherproofing, egress
- **Cumulative Damage**: Single-event dataset doesn't capture multi-event deterioration

---

## Summary Statistics Generated

### Missingness Report
Created `tornado_vulnerability_outputs/missingness_report.csv` with:
- Feature-by-feature missingness percentages
- Type classification (numeric vs. categorical)
- Sorted by missingness severity

### Age-Damage Crosstab
```
Age Category    Low    Significant    Undamaged
Pre-1880        5.7%   3.8%          90.6%
1880-1900       6.9%   19.6%         73.5%
1900-1920       4.9%   30.5%         64.6%
1920-1950       5.3%   14.9%         79.8%
```

---

## Files Modified
1. `tornado_vulnerability_paper_updated.tex` - All substantive changes
2. `tornado_refs.bib` - Added Mehta and Marshall citations
3. `analyze_missingness.py` - New analysis script (created)
4. `tornado_vulnerability_outputs/missingness_report.csv` - New data file

---

## Key Takeaways for User

### Strengths of Updates:
1. **Transparency**: Missingness now explicitly reported and discussed
2. **Mechanistic Depth**: Fenestration discussion now includes competing hypotheses with citations
3. **Honest Interpretation**: Age analysis acknowledges survivorship bias
4. **Practical Guidance**: Tiered framework provides actionable policy recommendations
5. **Preservation Realism**: Reversibility discussion now acknowledges material permanence

### Remaining Highlighted Items for User:
- `\hl{saanchi to come back and see if this needs to be moved above}`
- `\hl{add citation to our document, add saanchi papers}`
- `\hl{add citation to our own docs}`
- `\hl{make sure the ears are consistent everywhere}`
- `\hl{I think this reference is missing}`
- `\hl{add citation}`
- `\hl{cite Joe somewhere?}`
- `\hl{add career one}`

These are user-specific tasks that require domain knowledge or access to unpublished work.
