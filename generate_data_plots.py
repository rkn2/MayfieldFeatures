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
df_nash = pd.read_csv('updatedData/Nashville_Tornado_DataInput_Final_111425(in).csv')
df_qs = pd.read_csv('updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv', encoding='latin1')

# Add 'Event' column before merging
df_nash['Event'] = 'Nashville'
df_qs['Event'] = 'Quad State'

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

# Set style - Tufte/Minimalist (No Grid)
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.6) # Increased from 1.2
plt.rcParams['axes.grid'] = False

damage_order = ['Undamaged', 'Low', 'Significant']
damage_colors = {'Undamaged': '#2ecc71', 'Low': '#f1c40f', 'Significant': '#e74c3c'}
event_colors = {'Nashville': '#3498db', 'Quad State': '#e74c3c'}

# --- Plot 1: Year Built by Event ---
if 'year_built_u' in df.columns:
    plt.figure(figsize=(12, 8)) # Increased size
    sns.histplot(data=df, x='year_built_u', hue='event', multiple='stack', palette=event_colors, binwidth=10, edgecolor='white', linewidth=0.5)
    plt.title('Distribution of Year Built by Event', fontsize=18)
    plt.xlabel('Year Built', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xlim(1800, 2022)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/supp_year_built_by_event.png', dpi=300)
    plt.close()
    print("Generated supp_year_built_by_event.png")

# --- Plot 2: EF Rating Distribution (split by event)
# Convert EF ratings (including 'subEF') to numeric values (-1 for subEF)
ef_col_raw = 'tornado_ef'
if ef_col_raw in df.columns:
    def parse_ef(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s == 'subef':
            return -1
        s = s.replace('ef', '')
        try:
            return int(float(s))
        except:
            return np.nan
    df['ef_numeric'] = df[ef_col_raw].apply(parse_ef)
    # Create a display column where -1 is shown as 'sub'
    df['ef_display'] = df['ef_numeric'].replace({-1: 'sub'})
    plt.figure(figsize=(10, 8))  # Increased size
    ef_order = ['sub', 0, 1, 2, 3, 4]
    sns.countplot(data=df, x='ef_display', hue='event', order=ef_order, palette=event_colors, edgecolor='white', linewidth=0.5)
    plt.title('Distribution of EF Ratings by Event', fontsize=18)
    plt.xlabel('EF Rating', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title='Event')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/supp_ef_rating_dist.png', dpi=300)
    plt.close()
    print("Generated supp_ef_rating_dist.png")

# --- Plot 3: Damage by Event ---
plt.figure(figsize=(10, 8)) # Increased size
# Calculate percentages
ct = pd.crosstab(df['event'], df['Damage_Class'])
ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
ct_norm = ct_norm[damage_order] # Reorder

ax = ct_norm.plot(kind='bar', stacked=True, color=[damage_colors[x] for x in damage_order], figsize=(10, 8), width=0.7)
plt.title('Damage Distribution by Event', fontsize=18)
plt.xlabel('Event', fontsize=16)
plt.ylabel('Percentage', fontsize=16)
plt.legend(title='Damage Class', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=14)
plt.xticks(rotation=0, fontsize=14)
plt.yticks(fontsize=14)
sns.despine()
plt.tight_layout()
plt.savefig(f'{output_dir}/supp_damage_by_event.png', dpi=300)
plt.close()
print("Generated supp_damage_by_event.png")

# --- Plot 4: Year Built by Damage ---
if 'year_built_u' in df.columns:
    plt.figure(figsize=(12, 8)) # Increased size
    sns.boxplot(x='Damage_Class', y='year_built_u', data=df, order=damage_order, palette=damage_colors, linewidth=2)
    plt.title('Year Built Distribution by Damage Class', fontsize=18)
    plt.xlabel('Damage Class', fontsize=16)
    plt.ylabel('Year Built', fontsize=16)
    plt.ylim(1850, 2022) # Focus on relevant range
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/supp_year_built_by_damage.png', dpi=300)
    plt.close()
    print("Generated supp_year_built_by_damage.png")
