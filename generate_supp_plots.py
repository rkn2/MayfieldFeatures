import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = 'tornado_vulnerability_outputs'
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading data...")
df_nash = pd.read_excel('Nashville_Tornado_DataInput_Final_110725.xlsx')
df_qs = pd.read_csv('QuadState_Tornado_DataInputv2.csv', encoding='latin1')

# Normalize columns
def normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)
    return df

df_nash = normalize_cols(df_nash)
df_qs = normalize_cols(df_qs)
df = pd.concat([df_nash, df_qs], axis=0, ignore_index=True)

# Map target
def map_target(val):
    if pd.isna(val): return np.nan
    if val == 0: return 'Undamaged'
    if val == 1: return 'Low'
    if val >= 2: return 'Significant'
    return np.nan

df['Damage_Class'] = df['degree_of_damage_u'].apply(map_target)
df = df.dropna(subset=['Damage_Class'])

# Set style - Tufte/Minimalist
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.6) # Increased from 1.2
plt.rcParams['axes.grid'] = False

damage_order = ['Undamaged', 'Low', 'Significant']
damage_colors = {'Undamaged': '#2ecc71', 'Low': '#f1c40f', 'Significant': '#e74c3c'}

# --- Plot 1: MWFRS vs Damage ---
if 'mwfrs_u_wall' in df.columns:
    plt.figure(figsize=(12, 8)) # Increased size
    # Calculate counts
    ct = pd.crosstab(df['mwfrs_u_wall'], df['Damage_Class'])
    # Normalize to percentages
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    # Reorder columns
    ct_norm = ct_norm[damage_order]
    
    ax = ct_norm.plot(kind='bar', stacked=True, color=[damage_colors[x] for x in damage_order], figsize=(12, 8), width=0.8)
    plt.title('Damage Distribution by MWFRS Type', fontsize=18)
    plt.xlabel('MWFRS Type', fontsize=16)
    plt.ylabel('Percentage', fontsize=16)
    plt.legend(title='Damage Class', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/supp_mwfrs_damage.png', dpi=300)
    plt.close()
    print("Generated supp_mwfrs_damage.png")

# --- Plot 2: Occupancy vs Damage ---
if 'occupany_u' in df.columns:
    plt.figure(figsize=(12, 8)) # Increased size
    # Filter out rare occupancies (< 5 buildings) to keep plot clean
    counts = df['occupany_u'].value_counts()
    common_occupancies = counts[counts >= 5].index
    df_occ = df[df['occupany_u'].isin(common_occupancies)]
    
    ct = pd.crosstab(df_occ['occupany_u'], df_occ['Damage_Class'])
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_norm = ct_norm[damage_order]
    
    ax = ct_norm.plot(kind='bar', stacked=True, color=[damage_colors[x] for x in damage_order], figsize=(12, 8), width=0.8)
    plt.title('Damage Distribution by Occupancy Type', fontsize=18)
    plt.xlabel('Occupancy Type', fontsize=16)
    plt.ylabel('Percentage', fontsize=16)
    plt.legend(title='Damage Class', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/supp_occupancy_damage.png', dpi=300)
    plt.close()
    print("Generated supp_occupancy_damage.png")

from scipy.stats import mannwhitneyu

# --- Plot 3: Parapet Height vs Damage ---
if 'parapet_height_m' in df.columns:
    # Clean numeric data
    df['parapet_height_m'] = pd.to_numeric(df['parapet_height_m'], errors='coerce')
    
    # Calculate stats
    undamaged = df[df['Damage_Class'] == 'Undamaged']['parapet_height_m'].dropna()
    significant = df[df['Damage_Class'] == 'Significant']['parapet_height_m'].dropna()
    
    if len(undamaged) > 0 and len(significant) > 0:
        stat, p_val = mannwhitneyu(undamaged, significant)
        p_text = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    else:
        p_text = ""

    # Sample sizes
    counts = df['Damage_Class'].value_counts()
    labels = [f"{cls}\n(n={counts.get(cls, 0)})" for cls in damage_order]
    
    plt.figure(figsize=(10, 8)) # Increased size
    ax = sns.boxplot(x='Damage_Class', y='parapet_height_m', data=df, order=damage_order, palette=damage_colors, linewidth=2)
    plt.title('Parapet Height Distribution by Damage Class', fontsize=18)
    plt.xlabel('Damage Class', fontsize=16)
    plt.ylabel('Parapet Height (m)', fontsize=16)
    plt.xticks(ticks=range(3), labels=labels, fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add p-value annotation
    if p_text:
        y_max = df['parapet_height_m'].max()
        plt.text(1, y_max * 0.9, f"Undamaged vs. Significant:\n{p_text} (Mann-Whitney U)", 
                 ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/supp_parapet_damage.png', dpi=300)
    plt.close()
    print(f"Generated supp_parapet_damage.png with p-value: {p_val}")

    # --- Occupancy Breakdown (for text) ---
    print("\n--- Occupancy Breakdown ---")
    for occ in ['Residential', 'Business', 'Not in Use']: # Adjust based on actual values
        subset = df[df['occupany_u'] == occ] # Check spelling 'occupany_u'
        if not subset.empty:
            para_mean = subset['parapet_height_m'].mean()
            para_std = subset['parapet_height_m'].std()
            print(f"{occ}: Parapet Mean={para_mean:.2f}, Std={para_std:.2f}, n={len(subset)}")
