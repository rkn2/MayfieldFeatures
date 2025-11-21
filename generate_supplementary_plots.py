import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from replicate_analysis import load_and_preprocess_data, engineer_features

# Configuration
OUTPUT_DIR = 'tornado_vulnerability_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- DESIGN SYSTEM ---
# Set global style
sns.set_style("white") # Remove gridlines
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.titlesize': 16,
    'axes.grid': False # Ensure grid is off
})

# 1. Color Palettes (Consistent Semantics)
# Damage: Green (Safe) -> Yellow (Caution) -> Red (Danger)
DAMAGE_PALETTE = {
    'Undamaged': '#2ca02c',   # Green (Standard Tab10 Green)
    'Low': '#E1BE6A',         # Muted Gold
    'Significant': '#C44E52'  # Deep Red
}

# Events: Distinct categorical colors
EVENT_PALETTE = {
    'Nashville': '#1f77b4',   # Blue (Changed to avoid conflict with Green Damage)
    'QuadState': '#8172B3'    # Purple
}

# 2. Label Mappings (Clean English)
LABEL_MAP = {
    'year_built_u': 'Year Built',
    'dataset_source': 'Tornado Event',
    'target': 'Damage Severity',
    'roof_shape_u': 'Roof Shape',
    'ef_numeric': 'EF Rating',
    'count': 'Number of Buildings',
    'percentage': 'Percentage (%)'
}

def apply_labels(ax, title=None, x=None, y=None):
    """Helper to apply clean labels from the map."""
    if title: ax.set_title(title, fontweight='bold', pad=15)
    if x: ax.set_xlabel(LABEL_MAP.get(x, x))
    if y: ax.set_ylabel(LABEL_MAP.get(y, y))

def generate_plots():
    # 1. Load Data
    df = load_and_preprocess_data()
    df = engineer_features(df)
    
    print(f"Data loaded. Shape: {df.shape}")
    
    # Ensure categorical order for Damage
    damage_order = ['Undamaged', 'Low', 'Significant']
    
    # --- Plot 1: Year Built Distribution by Tornado Event ---
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df, 
        x='year_built_u', 
        hue='dataset_source', 
        multiple='stack', 
        bins=30, 
        palette=EVENT_PALETTE,
        edgecolor='white'
    )
    apply_labels(plt.gca(), title='Distribution of Year Built by Tornado Event', x='year_built_u', y='count')
    plt.xlim(1850, 2022)
    plt.legend(title='Tornado Event', labels=['Quad State', 'Nashville']) 
    sns.despine() # Remove top/right spines
    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/supp_year_built_by_event.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

    # --- Plot 2: Damage Distribution by Tornado Event ---
    # Calculate counts and percentages
    counts = df.groupby(['dataset_source', 'target']).size().reset_index(name='count')
    totals = df.groupby('dataset_source').size().reset_index(name='total')
    counts = counts.merge(totals, on='dataset_source')
    counts['percentage'] = counts['count'] / counts['total'] * 100
    
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=counts, 
        x='dataset_source', 
        y='percentage', 
        hue='target', 
        hue_order=damage_order, 
        palette=DAMAGE_PALETTE,
        edgecolor='white'
    )
    apply_labels(plt.gca(), title='Damage Class Distribution by Tornado Event', x='dataset_source', y='percentage')
    plt.legend(title='Damage Severity')
    sns.despine()
    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/supp_damage_by_event.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

    # --- Plot 3: Year Built by Damage Class (Boxplot) ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df, 
        x='target', 
        y='year_built_u', 
        order=damage_order, 
        palette=DAMAGE_PALETTE
    )
    apply_labels(plt.gca(), title='Year Built Distribution by Damage Severity', x='target', y='year_built_u')
    sns.despine()
    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/supp_year_built_by_damage.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

    # --- Plot 4: Roof Shape vs Damage (Stacked Bar) ---
    # Debug: Print roof shape counts
    print("Roof Shape Counts:\n", df['roof_shape_u'].value_counts())

    # Calculate counts directly to ensure control
    roof_counts = df.groupby(['roof_shape_u', 'target']).size().unstack(fill_value=0)
    
    # Filter for top 4 roof shapes to ensure we catch Hip if it's less frequent
    top_roofs = roof_counts.sum(axis=1).nlargest(4).index
    roof_counts = roof_counts.loc[top_roofs]
    
    # Normalize to percentages
    roof_props = roof_counts.div(roof_counts.sum(axis=1), axis=0)
    
    # Ensure all columns exist
    for col in damage_order:
        if col not in roof_props.columns:
            roof_props[col] = 0
    roof_props = roof_props[damage_order] # Reorder
    
    # Plot using Pandas/Matplotlib directly for reliable stacking
    ax = roof_props.plot(
        kind='bar', 
        stacked=True, 
        figsize=(10, 6), 
        color=[DAMAGE_PALETTE[x] for x in damage_order],
        edgecolor='white',
        width=0.8
    )
    
    apply_labels(ax, title='Proportion of Damage Levels by Roof Shape', x='roof_shape_u', y='percentage')
    ax.set_ylabel('Proportion')
    ax.legend(title='Damage Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    sns.despine()
    plt.tight_layout()
    
    save_path = f'{OUTPUT_DIR}/supp_roof_shape_damage.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()
    
    # --- Plot 5: EF Rating Distribution ---
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df, 
        x='ef_numeric', 
        hue='dataset_source', 
        palette=EVENT_PALETTE,
        edgecolor='white'
    )
    apply_labels(plt.gca(), title='Distribution of EF Ratings by Tornado Event', x='ef_numeric', y='count')
    plt.legend(title='Tornado Event')
    sns.despine()
    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/supp_ef_rating_dist.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    generate_plots()
