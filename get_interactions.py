import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# --- Load Data & Preprocess (Same as usual) ---
df_nash = pd.read_csv('updatedData/Nashville_Tornado_DataInput_Final_111425(in).csv')
df_qs = pd.read_csv('updatedData/Revised_QuadState_Tornado_DataInput_pub - Copy_120525.csv', encoding='latin1')

def normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)
    return df

df_nash = normalize_cols(df_nash)
df_qs = normalize_cols(df_qs)
df = pd.concat([df_nash, df_qs], axis=0, ignore_index=True)

def map_target(val):
    if pd.isna(val): return np.nan
    if val == 0: return 'Undamaged'
    if val == 1: return 'Low'
    if val >= 2: return 'Significant'
    return np.nan
df['target'] = df['degree_of_damage_u'].apply(map_target)
df = df.dropna(subset=['target'])

# Distance
required_coords = ['tornado_start_lat', 'tornado_start_long', 'tornado_end_lat', 'tornado_end_long', 'latitude', 'longitude']
if all(c in df.columns for c in required_coords):
    def dist_calc(px, py, x1, y1, x2, y2):
        mean_lat = np.mean([y1, y2, py])
        lat_scale = 111.0
        lon_scale = 111.0 * np.cos(np.radians(mean_lat))
        px_km, py_km = px * lon_scale, py * lat_scale
        x1_km, y1_km = x1 * lon_scale, y1 * lat_scale
        x2_km, y2_km = x2 * lon_scale, y2 * lat_scale
        dx = x2_km - x1_km
        dy = y2_km - y1_km
        if dx == 0 and dy == 0: return np.sqrt((px_km - x1_km)**2 + (py_km - y1_km)**2)
        t = ((px_km - x1_km) * dx + (py_km - y1_km) * dy) / (dx*dx + dy*dy)
        t = np.clip(t, 0, 1)
        return np.sqrt((px_km - (x1_km + t * dx))**2 + (py_km - (y1_km + t * dy))**2)

    df['distance_km'] = df.apply(lambda r: dist_calc(r['longitude'], r['latitude'], r['tornado_start_long'], r['tornado_start_lat'], r['tornado_end_long'], r['tornado_end_lat']), axis=1)

def clean_unknowns(val):
    if isinstance(val, str) and val.strip().lower() in ['un', 'unknown', 'n/a', 'na']: return np.nan
    return val
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(clean_unknowns)

numeric = ['number_stories', 'year_built_u', 'building_area_m2', 'buidling_height_m', 'wall_length_side', 'wall_length_front', 'wall_thickness', 'parapet_height_m', 'overhang_length_u', 'distance_km', 'wall_fenestration_per_n', 'wall_fenestration_per_s', 'wall_fenestration_per_e', 'wall_fenestration_per_w']
categorical = ['archetype', 'occupany_u', 'building_urban_setting', 'building_position_on_street', 'roof_shape_u', 'roof_slope_u', 'construction_type_u', 'mwfrs_u_wall', 'mwfrs_u_roof', 'structural_wall_system_u', 'foundation_type_u', 'wall_substrate_u', 'wall_cladding_u', 'roof_system_u', 'roof_substrate_type_u', 'roof_cover_u', 'retrofit_present_u', 'retrofit_type_u']
features = [f for f in numeric + categorical if f in df.columns and 'damage' not in f.lower()]

X = df[features].copy()
le = LabelEncoder()
y = le.fit_transform(df['target'])

# Preprocess
for c in X.columns:
    if c in numeric:
        if X[c].dtype == object:
            X[c] = X[c].replace('nan', np.nan)
            X[c] = X[c].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
        X[c] = pd.to_numeric(X[c], errors='coerce')
        X[c] = X[c].fillna(X[c].median())
    else:
        # Handle nan strings in categorical too if present
        X[c] = X[c].replace(np.nan, 'missing').replace('nan', 'missing')
        X[c] = X[c].astype(str)
X_enc = pd.get_dummies(X, columns=[c for c in categorical if c in X.columns], drop_first=True)

# Train RF
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_enc, y)

# --- SHAP Interaction Values ---
print("Calculating SHAP interaction values (this may take a moment)...")
explainer = shap.TreeExplainer(rf)
# Limit to a subset if needed, but 382 is small so full set is fine.
shap_interaction_values = explainer.shap_interaction_values(X_enc)

# shap_interaction_values shape: (Samples, Features, Features, Classes)
# Class 2 (Significant) is index 1 or 2.
# LE Classes: ['Low', 'Significant', 'Undamaged'] -> Indices: 0, 1, 2
sig_idx = 1 
shap_interaction_sig = shap_interaction_values[:, :, :, sig_idx]

# Calculate mean absolute interaction strength (off-diagonal)
# We want the strength of Feature A x Feature B
# The matrix is symmetric. We take the upper triangle.

mean_interaction = np.abs(shap_interaction_sig).mean(0) # Shape: (Features, Features)

# Create a DataFrame of interactions
interactions = []
feature_names = X_enc.columns
for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):
        val = mean_interaction[i, j] * 2 # Multiply by 2 because it's split between (i,j) and (j,i)
        interactions.append({
            'Feature 1': feature_names[i],
            'Feature 2': feature_names[j],
            'Interaction Strength': val
        })

int_df = pd.DataFrame(interactions).sort_values('Interaction Strength', ascending=False)
print("\nTop 15 Feature Interactions (Significant Damage):")
print(int_df.head(15))

# Also aggregate by parent feature for cleaner table
# Logic: Sum interaction strength for all pairs (Parent A_x, Parent B_y)
print("\n--- Aggregating by Parent Feature ---")
parent_interactions = {}

def get_parent(feat):
    for cat in categorical:
        if feat.startswith(cat + '_'):
            return cat
    return feat

for idx, row in int_df.iterrows():
    p1 = get_parent(row['Feature 1'])
    p2 = get_parent(row['Feature 2'])
    if p1 == p2: continue # Ignore self-interaction (main effect often bleeds here)
    
    # Sort to ensure (A, B) is same as (B, A)
    pair = tuple(sorted([p1, p2]))
    parent_interactions[pair] = parent_interactions.get(pair, 0) + row['Interaction Strength']

parent_int_df = pd.DataFrame([
    {'Interaction': f"{p[0]} x {p[1]}", 'Strength': s} 
    for p, s in parent_interactions.items()
]).sort_values('Strength', ascending=False)

print(parent_int_df.head(15))
