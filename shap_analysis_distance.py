import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import StratifiedKFold

# Configuration
RANDOM_STATE = 42
OUTPUT_DIR = 'tornado_vulnerability_outputs_damage_target' # Use the new specific folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data for SHAP analysis (Hazard-Neutral with Distance)...")

# Load data (same logic as before)
df_nash = pd.read_csv('updatedData/Nashville_Tornado_DataInput_Final_111425(in).csv')
df_qs = pd.read_csv('updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv', encoding='latin1')

def normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)
    return df

df_nash = normalize_cols(df_nash)
df_qs = normalize_cols(df_qs)
df = pd.concat([df_nash, df_qs], axis=0, ignore_index=True)

# Target
def map_target(val):
    if pd.isna(val): return np.nan
    if val == 0: return 'Undamaged'
    if val == 1: return 'Low'
    if val >= 2: return 'Significant'
    return np.nan

df['target'] = df['degree_of_damage_u'].apply(map_target)
df = df.dropna(subset=['target'])

# Distance Calculation
required_coords = ['tornado_start_lat', 'tornado_start_long', 'tornado_end_lat', 'tornado_end_long', 'latitude', 'longitude']
if all(c in df.columns for c in required_coords):
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def point_line_segment_distance(px, py, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return haversine_km(py, px, y1, x1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
        t = np.clip(t, 0, 1)
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return haversine_km(py, px, closest_y, closest_x)

    df['distance_km'] = df.apply(
        lambda row: point_line_segment_distance(
            row['longitude'], row['latitude'],
            row['tornado_start_long'], row['tornado_start_lat'],
            row['tornado_end_long'], row['tornado_end_lat']
        ), axis=1
    )

# Clean unknowns
def clean_unknowns(val):
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ['un', 'unknown', 'n/a', 'na']: return np.nan
    return val

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(clean_unknowns)

# Feature Sets (From replicate_analysis_damage_target.py logic)
numeric_features = [
    'number_stories', 'year_built_u', 'building_area_m2', 'buidling_height_m', 
    'wall_length_side', 'wall_length_front', 
    'wall_thickness', 'parapet_height_m', 'overhang_length_u',
    'wall_fenestration_per_n', 'wall_fenestration_per_s', 
    'wall_fenestration_per_e', 'wall_fenestration_per_w'
]
categorical_features = [
    'archetype', 'occupany_u', 'building_urban_setting', 'building_position_on_street', 
    'roof_shape_u', 'roof_slope_u', 'construction_type_u', 'mwfrs_u_wall', 
    'mwfrs_u_roof', 'structural_wall_system_u', 'foundation_type_u', 
    'wall_substrate_u', 'wall_cladding_u', 'roof_system_u', 'roof_substrate_type_u', 
    'roof_cover_u', 'retrofit_present_u', 'retrofit_type_u'
]
# Crucial: Include distance_km, Exclude ef_numeric
hazard_features = ['distance_km']

# Filter leakage
numeric_features = [f for f in numeric_features if 'damage' not in f.lower()]
categorical_features = [f for f in categorical_features if 'damage' not in f.lower()]

all_features = numeric_features + categorical_features + hazard_features
all_features = [f for f in all_features if f in df.columns]

# Prepare X, y
X = df[all_features].copy()
le = LabelEncoder()
y = le.fit_transform(df['target'])

# Preprocess
for col in X.columns:
    if col in numeric_features or col in hazard_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())
    else:
        # One-hot encode expects strings
        X[col] = X[col].astype(str).fillna('missing')

# One-hot encode
X_encoded = pd.get_dummies(X, columns=[c for c in categorical_features if c in X.columns], drop_first=True)
X_encoded = X_encoded.dropna(axis=1, how='all')

print(f"Feature shape: {X_encoded.shape}")

# Run SHAP
print("Running SHAP for Random Forest...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
all_shap_values = np.zeros((X_encoded.shape[0], X_encoded.shape[1], len(le.classes_)))

# Params matching replicate_analysis
rf_params = {
    'n_estimators': 200, 
    'class_weight': 'balanced_subsample', 
    'min_samples_leaf': 2, 
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

for fold, (train_idx, test_idx) in enumerate(skf.split(X_encoded, y)):
    print(f"Fold {fold+1}/5")
    X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
    y_train = y[train_idx]
    
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(model)
    shap_vals_fold = explainer.shap_values(X_test)
    
    # Handle list vs array output
    if isinstance(shap_vals_fold, list):
        shap_vals_fold_np = np.stack(shap_vals_fold, axis=-1)
    else:
        shap_vals_fold_np = shap_vals_fold
        
    all_shap_values[test_idx] = shap_vals_fold_np

# Plots
print("Generating Plots...")
# Need a fitted model for base_values (using full data)
model_full = RandomForestClassifier(**rf_params)
model_full.fit(X_encoded, y)
explainer_full = shap.TreeExplainer(model_full)

shap_obj = shap.Explanation(
    values=all_shap_values,
    base_values=explainer_full.expected_value,
    data=X_encoded.values,
    feature_names=X_encoded.columns
)

# Plot for Significant Damage feature contributions
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_obj[:, :, 2], max_display=20, show=False)
plt.title('SHAP Summary: Significant Damage (Class 2)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_beeswarm_class2.png', dpi=300, bbox_inches='tight')

# Plot for Undamaged
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_obj[:, :, 0], max_display=20, show=False)
plt.title('SHAP Summary: Undamaged (Class 0)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_beeswarm_class0.png', dpi=300, bbox_inches='tight')

print(f"SHAP done. Plots saved to {OUTPUT_DIR}")
