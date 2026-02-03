import pandas as pd

perm = pd.read_csv('tornado_vulnerability_outputs_engineered/permutation_importance.csv')
perm = perm[perm['Model'] == 'RandomForest']
mean_perm = perm.groupby('Feature')['Decrease_in_Accuracy'].mean().sort_values(ascending=False)

rfe = pd.read_csv('tornado_vulnerability_outputs_rfe/rfe_ranking.csv')
rfe_top = rfe[rfe['Support'] == True]['Feature'].tolist()

print("--- Top 10 Permutation Importance ---")
print(mean_perm.head(10))

print("\n--- RFE Selected Features (Ranking 1) ---")
print(rfe_top)

common = set(mean_perm.head(10).index) & set(rfe_top)
print(f"\nCommon features in both: {common}")
