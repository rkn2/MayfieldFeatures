import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import os

# Create output directory
output_dir = 'tornado_vulnerability_outputs'
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
print("Loading data...")
df_nash = pd.read_excel('Nashville_Tornado_DataInput_Final_110725.xlsx')
df_qs = pd.read_csv('QuadState_Tornado_DataInputv2.csv', encoding='latin1')

def normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)
    return df

df_nash = normalize_cols(df_nash)
df_qs = normalize_cols(df_qs)
df = pd.concat([df_nash, df_qs], axis=0, ignore_index=True)

# 2. Preprocessing
# Map target
def map_target(val):
    if pd.isna(val): return np.nan
    if val == 0: return 0 # Undamaged
    if val == 1: return 1 # Low
    if val >= 2: return 2 # Significant
    return np.nan

df['target'] = df['degree_of_damage_u'].apply(map_target)
df = df.dropna(subset=['target'])

# Select features (Hazard-Neutral set + Parapet)
# We need to make sure parapet_height_m is numeric
df['parapet_height_m'] = pd.to_numeric(df['parapet_height_m'], errors='coerce')

# Define features to use for the model (simplified set for SHAP focus)
feature_cols = [
    'parapet_height_m', 'mwfrs_u_wall', 'occupany_u', 'foundation_type_u', 
    'wall_thickness', 'wall_length_front', 'wall_cladding_u', 
    'wall_fenestration_per_w', 'roof_slope_u', 'construction_type_u',
    'building_urban_setting', 'retrofit_type_u', 'structural_wall_system_u',
    'distance_km'
]

# Filter to columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].copy()
y = df['target']

# Handle categorical encoding
categorical_cols = X.select_dtypes(include=['object']).columns
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_cols] = encoder.fit_transform(X[categorical_cols].astype(str))

# Impute missing numeric values
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 3. Train Model
print("Training RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X, y)

# 4. Compute SHAP
print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print(f"SHAP values type: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"SHAP values list length: {len(shap_values)}")
    print(f"SHAP values[0] shape: {shap_values[0].shape}")
else:
    print(f"SHAP values shape: {shap_values.shape}")
print(f"X shape: {X.shape}")

# 5. Generate Dependence Plot for Parapet Height (Class 2: Significant Damage)
# We want to see how parapet height affects the risk of Significant Damage
class_idx = 2 

# List of features to plot
features_to_plot = [
    ('parapet_height_m', 'Parapet Height'),
    ('wall_fenestration_per_w', 'Fenestration % (West)'),
    ('distance_km', 'Distance to Track')
]

for feat_col, feat_name in features_to_plot:
    if feat_col in X.columns:
        plt.figure(figsize=(10, 7))
        # Plot
        shap.dependence_plot(
            feat_col, 
            shap_values[:, :, class_idx], 
            X, 
            interaction_index=None, 
            show=False,
            alpha=0.7
        )
        
        # Add threshold line for fenestration
        if feat_col == 'wall_fenestration_per_w':
            plt.axvline(x=20, color='red', linestyle='--', linewidth=2, label='20% Threshold')
            plt.legend()

        plt.title(f'SHAP Dependence: {feat_col} (Significant Damage)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_dependence_{feat_col}.png')
        plt.close()
        print(f"Generated {output_dir}/shap_dependence_{feat_col}.png")
