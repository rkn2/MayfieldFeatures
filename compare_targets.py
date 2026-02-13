import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Load permutation importance from both analyses
print("Loading permutation importance data...")
perm_degree = pd.read_csv('tornado_vulnerability_outputs_engineered/permutation_importance.csv')
perm_status = pd.read_csv('tornado_vulnerability_outputs_status_target/permutation_importance.csv')

# Filter to RandomForest only for consistency
perm_degree_rf = perm_degree[perm_degree['Model'] == 'RandomForest']
perm_status_rf = perm_status[perm_status['Model'] == 'RandomForest']

# Calculate mean importance across folds
mean_degree = perm_degree_rf.groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False)
mean_status = perm_status_rf.groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False)

print("\n=== TOP 15 FEATURES: degree_of_damage_u ===")
print(mean_degree.head(15))

print("\n=== TOP 15 FEATURES: status_u ===")
print(mean_status.head(15))

# Get common features
common_features = list(set(mean_degree.index) & set(mean_status.index))
common_features = [f for f in common_features if f != 'random_noise']

# Create comparison dataframe
comparison = pd.DataFrame({
    'Feature': common_features,
    'degree_of_damage_u_importance': [mean_degree.get(f, 0) for f in common_features],
    'status_u_importance': [mean_status.get(f, 0) for f in common_features],
    'degree_of_damage_u_rank': [list(mean_degree.index).index(f) + 1 if f in mean_degree.index else 999 for f in common_features],
    'status_u_rank': [list(mean_status.index).index(f) + 1 if f in mean_status.index else 999 for f in common_features]
})

comparison['rank_difference'] = abs(comparison['degree_of_damage_u_rank'] - comparison['status_u_rank'])
comparison = comparison.sort_values('degree_of_damage_u_rank')

# Calculate Spearman correlation
rho, p_value = spearmanr(comparison['degree_of_damage_u_rank'], comparison['status_u_rank'])
print(f"\n=== RANK CORRELATION ===")
print(f"Spearman's ρ: {rho:.3f}")
print(f"p-value: {p_value:.4f}")

# Save comparison table
comparison.to_csv('tornado_vulnerability_outputs_status_target/feature_comparison.csv', index=False)
print("\nComparison table saved to: tornado_vulnerability_outputs_status_target/feature_comparison.csv")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Side-by-side top 10
top_10_degree = mean_degree.head(10)
top_10_status = mean_status.head(10)

ax1 = axes[0]
y_pos = np.arange(len(top_10_degree))
ax1.barh(y_pos, top_10_degree.values, color='steelblue', alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_10_degree.index)
ax1.invert_yaxis()
ax1.set_xlabel('Mean Decrease in Accuracy')
ax1.set_title('Top 10 Features: degree_of_damage_u')
ax1.grid(axis='x', alpha=0.3)

ax2 = axes[1]
y_pos = np.arange(len(top_10_status))
ax2.barh(y_pos, top_10_status.values, color='coral', alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top_10_status.index)
ax2.invert_yaxis()
ax2.set_xlabel('Mean Decrease in Accuracy')
ax2.set_title('Top 10 Features: status_u')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('tornado_vulnerability_outputs_status_target/target_comparison_top10.png', dpi=300)
print("Visualization saved to: tornado_vulnerability_outputs_status_target/target_comparison_top10.png")

# Plot 2: Scatter plot of ranks
fig, ax = plt.subplots(figsize=(10, 10))
scatter_data = comparison[comparison['degree_of_damage_u_rank'] <= 20]
ax.scatter(scatter_data['degree_of_damage_u_rank'], scatter_data['status_u_rank'], 
           s=100, alpha=0.6, edgecolors='black')

# Add diagonal line
max_rank = max(scatter_data['degree_of_damage_u_rank'].max(), scatter_data['status_u_rank'].max())
ax.plot([0, max_rank], [0, max_rank], 'r--', alpha=0.5, label='Perfect Agreement')

# Annotate key features
for idx, row in scatter_data.iterrows():
    if row['degree_of_damage_u_rank'] <= 10 or row['status_u_rank'] <= 10:
        ax.annotate(row['Feature'], 
                   (row['degree_of_damage_u_rank'], row['status_u_rank']),
                   fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Rank (degree_of_damage_u)', fontsize=12)
ax.set_ylabel('Rank (status_u)', fontsize=12)
ax.set_title(f'Feature Rank Comparison\nSpearman ρ = {rho:.3f}, p = {p_value:.4f}', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('tornado_vulnerability_outputs_status_target/rank_correlation_scatter.png', dpi=300)
print("Scatter plot saved to: tornado_vulnerability_outputs_status_target/rank_correlation_scatter.png")

# Identify features with large rank changes
print("\n=== FEATURES WITH LARGEST RANK CHANGES ===")
large_changes = comparison.nlargest(10, 'rank_difference')[['Feature', 'degree_of_damage_u_rank', 'status_u_rank', 'rank_difference']]
print(large_changes)

# Check if Distance remains top predictor
print("\n=== DISTANCE RANKING ===")
if 'distance_km' in comparison['Feature'].values:
    dist_row = comparison[comparison['Feature'] == 'distance_km'].iloc[0]
    print(f"degree_of_damage_u: Rank {dist_row['degree_of_damage_u_rank']}")
    print(f"status_u: Rank {dist_row['status_u_rank']}")
else:
    print("distance_km not found in features")

print("\n=== ANALYSIS COMPLETE ===")
