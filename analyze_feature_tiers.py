import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# Load results
df = pd.read_csv('tornado_vulnerability_outputs_engineered/permutation_importance.csv')

# Filter for RandomForest and the correct setting
df = df[df['Model'] == 'RandomForest']
# Assumption: there is only one setting, or pick the first one
setting = df['Setting'].unique()[0]
df = df[df['Setting'] == setting]

# Calculate mean importance to establish ranking
mean_imp = df.groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False)
sorted_features = mean_imp.index.tolist()

print(f"--- Top 15 Features Ranking (Mean Decrease in Accuracy) ---")
print(mean_imp.head(15))
print("\n--- Statistical Tier Analysis (Pairwise Wilcoxon vs Next Ranked) ---")

tiers = []
current_tier = 1
tiers.append({'Tier': current_tier, 'Feature': sorted_features[0], 'Mean': mean_imp[sorted_features[0]]})

# Iterate and compare i with i+1
for i in range(len(sorted_features) - 1):
    feat_a = sorted_features[i]
    feat_b = sorted_features[i+1]
    
    # Get the 25 fold values for each feature
    # Ensure they are aligned by fold!
    # The csv should have Fold column.
    
    vals_a = df[df['Feature'] == feat_a].sort_values('Fold')['Decrease_in_Accuracy'].values
    vals_b = df[df['Feature'] == feat_b].sort_values('Fold')['Decrease_in_Accuracy'].values
    
    # Check alignment
    if len(vals_a) != len(vals_b):
        print(f"Error: Length mismatch for {feat_a} and {feat_b}")
        continue
        
    # Wilcoxon test
    # Null hypothesis: the distribution of differences is symmetric about zero (no difference)
    # If p < 0.05, they are significantly different -> New Tier
    # If p >= 0.05, they are separate tiers? No, they are "Statistical Ties" -> Same Tier
    
    # Handle zero differences issue in Wilcoxon? Scipy handles it.
    # Note: if all diffs are zero, it throws error or similar.
    
    try:
        stat, p = wilcoxon(vals_a - vals_b, alternative='two-sided')
    except ValueError:
        # Happens if all differences are zero
        p = 1.0
    
    is_diff = p < 0.05
    
    print(f"{feat_a} vs {feat_b}: p={p:.4f} -> {'Different' if is_diff else 'Equivalent'}")
    
    if is_diff:
        current_tier += 1
        
    tiers.append({'Tier': current_tier, 'Feature': feat_b, 'Mean': mean_imp[feat_b]})
    
    # Stop after top 20 or so to avoid noise
    if i >= 19:
        break

print("\n--- Final Tiers ---")
tier_df = pd.DataFrame(tiers)
print(tier_df)
