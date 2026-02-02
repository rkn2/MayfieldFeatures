import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC

# Load data (abbreviated for speed)
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

# Distance
required_coords = ['tornado_start_lat', 'tornado_start_long', 'tornado_end_lat', 'tornado_end_long', 'latitude', 'longitude']
if all(c in df.columns for c in required_coords):
    def point_line_segment_distance(px, py, x1, y1, x2, y2):
        mean_lat = np.mean([y1, y2, py])
        lat_scale = 111.0
        lon_scale = 111.0 * np.cos(np.radians(mean_lat))
        px_km, py_km = px * lon_scale, py * lat_scale
        x1_km, y1_km = x1 * lon_scale, y1 * lat_scale
        x2_km, y2_km = x2 * lon_scale, y2 * lat_scale
        dx = x2_km - x1_km
        dy = y2_km - y1_km
        if dx == 0 and dy == 0:
            return np.sqrt((px_km - x1_km)**2 + (py_km - y1_km)**2)
        t = ((px_km - x1_km) * dx + (py_km - y1_km) * dy) / (dx*dx + dy*dy)
        t = np.clip(t, 0, 1)
        closest_x = x1_km + t * dx
        closest_y = y1_km + t * dy
        return np.sqrt((px_km - closest_x)**2 + (py_km - closest_y)**2)

    df['distance_km'] = df.apply(
        lambda row: point_line_segment_distance(
            row['longitude'], row['latitude'],
            row['tornado_start_long'], row['tornado_start_lat'],
            row['tornado_end_long'], row['tornado_end_lat']
        ), axis=1
    )

def clean_unknowns(val):
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ['un', 'unknown', 'n/a', 'na']: return np.nan
    return val

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(clean_unknowns)

# Features
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
hazard_features = ['distance_km'] # Only distance

# Filter leakage
numeric_features = [f for f in numeric_features if 'damage' not in f.lower()]
categorical_features = [f for f in categorical_features if 'damage' not in f.lower()]

all_features = numeric_features + categorical_features + hazard_features
all_features = [f for f in all_features if f in df.columns]

X = df[all_features].copy()
y = df['target'] # Keep as string for report labels

# Preprocess
for col in X.columns:
    if col in numeric_features or col in hazard_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].astype(str).fillna('missing')

# Encode cats for SMOTE
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
cat_subset = [c for c in all_features if c in categorical_features]
X[cat_subset] = enc.fit_transform(X[cat_subset])
cat_idxs = [X.columns.get_loc(c) for c in cat_subset]

# Run 5-fold CV and average reports? Or just one robust report?
# The paper says "averaged over 25 folds". I'll approximate with 5 folds to get the numbers.
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

y_true_all = []
y_pred_all = []

print("Running CV...")
for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # SMOTE
    try:
        smote = SMOTENC(categorical_features=cat_idxs, random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    except:
        X_res, y_res = X_train, y_train
        
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', min_samples_leaf=2, random_state=42)
    clf.fit(X_res, y_res)
    
    y_pred = clf.predict(X_test)
    
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)

print("\n--- Classification Report ---")
print(classification_report(y_true_all, y_pred_all, digits=3))
