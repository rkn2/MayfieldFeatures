import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid", context="talk")

def load_and_preprocess_data():
    print("Loading data...")
    df_nash = pd.read_csv('updatedData/Nashville_Tornado_DataInput_Final_111425(in).csv')
    df_qs = pd.read_csv('updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv', encoding='latin1')
    
    # Normalize columns
    def normalize_cols(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)
        return df

    df_nash = normalize_cols(df_nash)
    df_qs = normalize_cols(df_qs)
    
    # Combine
    df = pd.concat([df_nash, df_qs], axis=0, ignore_index=True)
    return df

def map_degree_target(val):
    if pd.isna(val): return np.nan
    if val == 0: return 'Undamaged'
    if val == 1: return 'Low'
    if val >= 2: return 'Significant'
    return np.nan

def map_status_target(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s == 'undamaged': return 'Undamaged'
    if s == 'minor': return 'Low'
    if s in ['moderate', 'severe', 'destroyed']: return 'Significant'
    return np.nan

def generate_comparison_plot(df, output_dir):
    print("Generating comparison plot...")
    
    # Create target columns
    df['Degree Target'] = df['degree_of_damage_u'].apply(map_degree_target)
    df['Status Target'] = df['status_u'].apply(map_status_target)
    
    # Drop rows where either target is missing
    df_plot = df.dropna(subset=['Degree Target', 'Status Target'])
    
    print(f"Plotting with {len(df_plot)} samples.")
    
    # Order for plotting
    order = ['Undamaged', 'Low', 'Significant']
    
    # Melt dataframe for easier plotting with seaborn
    df_melted = pd.melt(df_plot[['Degree Target', 'Status Target']], 
                        var_name='Target Type', value_name='Damage Class')
    
    plt.figure(figsize=(10, 6))
    
    # Create countplot with hue
    ax = sns.countplot(data=df_melted, x='Damage Class', hue='Target Type', order=order, palette='muted')
    
    plt.title('Comparison of Damage Class Distribution\n(3-Class Structure)', fontsize=16)
    plt.xlabel('Damage Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(title='Target Variable')
    
    # Add value labels
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.text(p.get_x() + p.get_width()/2., height + 3,
                    f'{int(height)}', ha="center", fontsize=10)
            
    plt.tight_layout()
    plt.savefig(f'{output_dir}/target_comparison_histogram.png', dpi=300)
    print(f"Plot saved to {output_dir}/target_comparison_histogram.png")

if __name__ == "__main__":
    output_dir = 'tornado_vulnerability_outputs_status_target'
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_and_preprocess_data()
    generate_comparison_plot(df, output_dir)
