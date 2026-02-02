import seaborn as sns
import matplotlib.pyplot as plt
import os
from replicate_analysis_damage_target import load_and_preprocess_data, engineer_features

# Use the new folder to keep new results together
OUTPUT_DIR = 'tornado_vulnerability_outputs_damage_target' 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Consistent palette
DAMAGE_PALETTE = {
    'Undamaged': '#2ca02c',
    'Low': '#E1BE6A',
    'Significant': '#C44E52'
}
damage_order = ['Undamaged', 'Low', 'Significant']

def generate():
    # Load data using existing pipeline (ensures distance_km is calculated)
    df = load_and_preprocess_data()
    df = engineer_features(df)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("white")
    sns.boxplot(
        data=df, 
        x='target', 
        y='distance_km', 
        order=damage_order, 
        palette=DAMAGE_PALETTE
    )
    plt.title('Distance from Tornado Path by Damage Severity', fontweight='bold')
    plt.xlabel('Damage Severity')
    plt.ylabel('Distance to Tornado Centerline (km)')
    sns.despine()
    plt.tight_layout()
    
    save_path = f'{OUTPUT_DIR}/supp_dist_damage_.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    generate()
